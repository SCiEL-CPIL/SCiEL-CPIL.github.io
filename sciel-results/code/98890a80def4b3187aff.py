"""Rewritten program: branch-and-bound MIS on 2-shadow conflict graph with matching upper bound."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    DEADLINE = 85.0
    from math import lcm
    from itertools import combinations

    n = 128

    # Precompute LCM matrix
    lcm_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_mat[i][j] = lcm(i, j)
            lcm_mat[j][i] = lcm_mat[i][j]

    # Enumerate every equal-LCM triple (a < b < c) in [1,128]
    triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_mat[a][b]
            for c in range(b + 1, n + 1):
                if lcm_mat[a][c] == lab and lcm_mat[b][c] == lab:
                    triples.append((a - 1, b - 1, c - 1))  # 0-indexed
    m = len(triples)

    # Build 2-shadow conflict graph: edge (i,j) if i,j co-occur in some triple
    # Use bitsets for efficiency
    adj = [0] * n  # bitset of neighbors
    for a, b, c in triples:
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)

    # Greedy independent set heuristic
    def greedy_is():
        remaining = (1 << n) - 1
        selected = []
        while remaining:
            # Pick vertex with min degree in remaining
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v < 0:
                break
            selected.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return selected

    best_set = greedy_is()
    best_size = len(best_set)

    def is_valid(S):
        S_set = set(S)
        for a, b, c in triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True

    # Verify greedy
    if not is_valid(best_set):
        # Fallback: pure greedy adding
        best_set = []
        best_set_set = set()
        for x in range(n):
            ok = True
            for a, b in combinations(best_set, 2):
                if lcm_mat[a+1][b+1] == lcm_mat[a+1][x+1] == lcm_mat[b+1][x+1]:
                    ok = False
                    break
            if ok:
                best_set.append(x)
                best_set_set.add(x)
        best_size = len(best_set)

    # Branch and bound on the conflict graph for MIS
    # Upper bound: for remaining vertices R, alpha(R) <= |R| - max_matching(R)
    # But computing max matching each node is expensive. Use greedy matching as lower bound on matching.
    
    # Better: use a simple but effective bound. For a vertex v in R, either v is in the independent set
    # (then remove v and its neighbors) or v is not (remove v). Upper bound: max of the two branches.
    # Prune if current_size + |R| <= best_size.
    # Tighter: current_size + greedy_is_size(R) <= best_size -> prune.

    import sys
    sys.setrecursionlimit(10000)

    # Precompute for fast greedy IS on a bitset
    def greedy_is_bitset(remaining):
        selected = 0
        count = 0
        rem = remaining
        while rem:
            best_v = -1
            best_deg = n + 1
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (adj[v] & rem).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v < 0:
                break
            selected |= (1 << best_v)
            count += 1
            rem &= ~(1 << best_v)
            rem &= ~adj[best_v]
        return selected, count

    # Branch and bound
    # State: remaining bitset, current selected bitset, current size
    # We'll use a stack-based DFS with ordering by potential

    # For ordering: try vertices that are "critical" first
    # Use a simple recursive B&B with memoization on remaining set (but 2^128 is too large, so no memo)
    
    # Instead, use a greedy-guided B&B: at each node, pick a vertex, branch include/exclude
    # Bound: current_size + greedy_is_count(remaining)
    
    # To make it fast, we'll limit depth and use time checks
    
    def bb(remaining, current_size, selected_bits):
        nonlocal best_set, best_size
        if time.time() - start > DEADLINE:
            return
        if remaining == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = [i for i in range(n) if (selected_bits >> i) & 1]
            return
        
        # Upper bound: current_size + popcount(remaining)
        if current_size + remaining.bit_count() <= best_size:
            return
        
        # Tighter bound: greedy IS on remaining
        _, gcount = greedy_is_bitset(remaining)
        if current_size + gcount <= best_size:
            return
        
        # Pick a vertex to branch on: choose one with high degree in remaining
        temp = remaining
        best_v = -1
        best_deg = -1
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v < 0:
            return
        
        # Branch 1: include best_v
        new_remaining = remaining & ~(1 << best_v) & ~adj[best_v]
        bb(new_remaining, current_size + 1, selected_bits | (1 << best_v))
        
        if time.time() - start > DEADLINE:
            return
        
        # Branch 2: exclude best_v
        new_remaining = remaining & ~(1 << best_v)
        bb(new_remaining, current_size, selected_bits)

    # Start B&B
    all_bits = (1 << n) - 1
    # First try to improve with B&B
    if time.time() - start < DEADLINE - 5:
        bb(all_bits, 0, 0)

    # Convert to 1-indexed
    result = [i + 1 for i in range(n) if (best_set[i] if isinstance(best_set, list) and len(best_set) == n else False)]
    
    # Safer: rebuild from best_set
    if isinstance(best_set, list):
        if all(isinstance(x, int) and 0 <= x < n for x in best_set):
            result = [x + 1 for x in best_set]
        else:
            result = [i + 1 for i in best_set]
    
    # Final validity check
    result_set = set(result)
    valid = True
    for a, b, c in triples:
        if (a+1) in result_set and (b+1) in result_set and (c+1) in result_set:
            valid = False
            break
    
    if not valid or len(result) == 0:
        # Fallback to greedy
        result = [x + 1 for x in greedy_is()]
    
    return result
# EVOLVE-BLOCK-END
