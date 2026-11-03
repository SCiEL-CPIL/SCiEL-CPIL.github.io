"""Fresh design: greedy distance-2 independent-set cover + aggressive local search."""
# EVOLVE-BLOCK-START
import sys
import time
import random
import itertools
from collections import deque
from heapq import heappush, heappop

import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0          # safety margin before the 120 s kill
    rng = random.Random(12345)

    if q <= 0 or n <= 0:
        return []

    # Trivial / small cases
    if radius >= n:
        return [[0] * n]
    if n == 1:
        return [[i] for i in range(q)]
    if q == 2:
        # binary: radius 1 -> parity code (2^(n-1) words) is optimal
        if radius == 1:
            out = []
            for i in range(1 << (n - 1)):
                v = i
                parity = bin(i).count("1") & 1
                bits = [((i >> (n - 1 - k)) & 1) for k in range(n - 1)]
                bits.append(parity)
                out.append(bits)
            return out
        # fallback: general path below

    # ---------- Build the distance-2 graph (edges = pairs at Hamming distance 2) ----------
    # A codeword covers all words within distance `radius`.  Pairs at distance 2 are the
    # classic removable candidates: their union of spheres is almost the same as a single
    # well-placed sphere, so a code that avoids such pairs is easier to shrink.
    #
    # We generate the distance-2 graph implicitly (never materialise all 3^n edges).
    # For q=3, n=6 there are 729 vertices; the graph has 540 edges.

    N = q ** n

    # Map integer index <-> vector (base q, little-endian)
    def idx_to_vec(idx):
        v = [0] * n
        x = idx
        for k in range(n):
            v[k] = x % q
            x //= q
        return v

    # Precompute, for each position i and value a, the list of (pos, newval) that change
    # coordinate i from a to newval (newval != a).  We use this to enumerate distance-2
    # neighbours quickly.
    # For a vertex v, its distance-2 neighbours are: pick two positions i<j, change both.
    # That's C(n,2)*(q-1)^2 neighbours per vertex.  For q=3,n=6: 15*4 = 60 neighbours.

    def distance2_neighbours(idx):
        v = idx_to_vec(idx)
        nbrs = []
        for i in range(n):
            vi = v[i]
            for j in range(i + 1, n):
                vj = v[j]
                for ai in range(q):
                    if ai == vi:
                        continue
                    for aj in range(q):
                        if aj == vj:
                            continue
                        # new index: change coord i to ai, coord j to aj
                        ni = idx + (ai - vi) * (q ** i) + (aj - vj) * (q ** j)
                        nbrs.append(ni)
        return nbrs

    # Build adjacency as a list of lists (symmetric).  For 729 vertices this is fine.
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in distance2_neighbours(i):
            if j > i:
                adj[i].append(j)
                adj[j].append(i)

    # ---------- Greedy: build a maximal independent set that is also a cover ----------
    # We want a set S such that:
    #   (a) S is an independent set in the distance-2 graph (no two codewords at dist 2)
    #   (b) every vertex is within Hamming distance `radius` of some codeword in S
    #
    # Greedy strategy: repeatedly pick the vertex that (1) is uncovered or covers the most
    # uncovered vertices, (2) is not adjacent (dist 2) to any already-chosen vertex.
    #
    # For efficiency we maintain:
    #   covered: bool array, whether vertex is within radius of some chosen codeword
    #   forbidden: bool array, whether vertex is at distance 2 from some chosen codeword
    #
    # Precompute the closed ball (radius `radius`) around each vertex for coverage checks.

    def hamming_dist_vec(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Precompute closed balls
    all_vecs = [idx_to_vec(i) for i in range(N)]
    balls = []
    for i in range(N):
        vi = all_vecs[i]
        ball = [j for j in range(N) if hamming_dist_vec(vi, all_vecs[j]) <= radius]
        balls.append(ball)

    covered = np.zeros(N, dtype=bool)
    forbidden = np.zeros(N, dtype=bool)
    chosen = []

    # Precompute "covers" count: how many currently-uncovered vertices each candidate covers
    # We'll recompute incrementally.
    remaining = set(range(N))

    while remaining:
        if time.time() > deadline - 30.0:
            break
        # Score each remaining candidate: number of uncovered vertices in its ball,
        # but only consider non-forbidden candidates.
        best = -1
        best_score = -1
        # To keep this fast, iterate over remaining and skip forbidden
        for cand in remaining:
            if forbidden[cand]:
                continue
            cnt = 0
            for u in balls[cand]:
                if not covered[u]:
                    cnt += 1
            if cnt > best_score:
                best_score = cnt
                best = cand
        if best < 0:
            # All remaining are forbidden; pick any remaining (relax independence)
            best = next(iter(remaining))
        chosen.append(best)
        remaining.discard(best)
        # Update covered
        for u in balls[best]:
            covered[u] = True
        # Update forbidden: mark all distance-2 neighbours of best
        for nb in adj[best]:
            forbidden[nb] = True

    # If we broke early, ensure coverage by adding uncovered-centre codewords
    if not covered.all():
        # Add codewords at uncovered vertices (or their centres) until covered
        while not covered.all():
            # find an uncovered vertex
            unc = np.where(~covered)[0]
            if len(unc) == 0:
                break
            # pick the uncovered vertex that covers the most other uncovered vertices
            best = -1
            best_cnt = -1
            for cand in range(N):
                cnt = 0
                for u in balls[cand]:
                    if not covered[u]:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = cand
            if best < 0:
                best = int(unc[0])
            chosen.append(best)
            for u in balls[best]:
                covered[u] = True
            if time.time() > deadline - 10.0:
                break

    codewords = [all_vecs[i] for i in chosen]

    # ---------- Local search to shrink the code ----------
    # Represent coverage with a count array: cover_count[u] = number of codewords covering u.
    # A codeword c can be removed if min_{u in ball(c)} cover_count[u] >= 2.

    def vec_to_idx(vec):
        idx = 0
        for k in range(n):
            idx += vec[k] * (q ** k)
        return idx

    def build_cover_count(cw_list):
        cc = np.zeros(N, dtype=np.int32)
        for c in cw_list:
            idx = vec_to_idx(c)
            for u in balls[idx]:
                cc[u] += 1
        return cc

    def can_remove(idx, cc):
        # True if removing codeword at idx keeps all vertices covered
        for u in balls[idx]:
            if cc[u] <= 1:
                return False
        return True

    def do_remove(idx, cc):
        for u in balls[idx]:
            cc[u] -= 1

    def do_add(idx, cc):
        for u in balls[idx]:
            cc[u] += 1

    cc = build_cover_count(codewords)
    current_indices = set(vec_to_idx(c) for c in codewords)

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # --- Pass 1: Single removals ---
        i = 0
        while i < len(codewords) and time.time() < deadline - 5.0:
            idx = vec_to_idx(codewords[i])
            if can_remove(idx, cc):
                do_remove(idx, cc)
                codewords.pop(i)
                current_indices.discard(idx)
                improved = True
                continue  # don't increment i
            i += 1
        
        if improved:
            continue

        # --- Pass 2: 1-swap (remove one, add another) ---
        swapped = False
        order = list(range(len(codewords)))
        rng.shuffle(order)
        
        for ri in order:
            if time.time() > deadline - 5.0:
                break
            
            ridx = vec_to_idx(codewords[ri])
            
            # Tentatively remove
            do_remove(ridx, cc)
            
            # Check if coverage is broken
            uncovered = []
            for u in balls[ridx]:
                if cc[u] == 0:
                    uncovered.append(u)
            
            if not uncovered:
                # Pure removal
                codewords.pop(ri)
                current_indices.discard(ridx)
                improved = True
                swapped = True
                break
            
            # We need to add a codeword that covers all `uncovered` vertices.
            # First try to find an independent replacement.
            found_independent = False
            for cand in range(N):
                if cand in current_indices:
                    continue
                
                # Check distance-2 constraint
                ok = True
                for nb in adj[cand]:
                    if nb in current_indices:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check coverage of uncovered vertices
                covers_all = True
                for u in uncovered:
                    if u not in balls[cand]:
                        covers_all = False
                        break
                
                if covers_all:
                    best_add = cand
                    found_independent = True
                    break
            
            if found_independent:
                do_add(best_add, cc)
                codewords[ri] = all_vecs[best_add]
                current_indices.add(best_add)
                swapped = True
                improved = True
                break
            
            else:
                # Try any replacement that restores coverage.
                for cand in range(N):
                    if cand in current_indices:
                        continue
                    
                    covers_all = True
                    for u in uncovered:
                        if u not in balls[cand]:
                            covers_all = False
                            break
                    
                    if covers_all:
                        best_add = cand
                        break
                
                if best_add != -1:
                    do_add(best_add, cc)
                    codewords[ri] = all_vecs[best_add]
                    current_indices.add(best_add)
                    swapped = True
                    improved = True
                    break
                else:
                    # Revert
                    do_add(ridx, cc)

        if improved:
            continue

        # If no improvements, break
        break

    # ---------- Final verification ----------
    final_cc = build_cover_count(codewords)
    if not final_cc.all():
        for u in range(N):
            if final_cc[u] == 0:
                codewords.append(all_vecs[u])
                for w in balls[u]:
                    final_cc[w] += 1

    # Ensure distinctness
    seen = set()
    unique = []
    for cw in codewords:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique.append(cw)

    return unique
# EVOLVE-BLOCK-END
