"""Portfolio SEARCH approach for K_3(6,1): greedy + local search + perturbation + flow-based refinement."""
import itertools
import random
import time
import sys

# Precompute for the small space 3^6 = 729.
_SPACE = [3] * 6  # length n placeholder

def _build_space(q, n, radius):
    # Return:
    #   words: list of tuples, length q**n, in lex order
    #   sphere: list of lists; sphere[i] = list of indices within Hamming distance radius of words[i]
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    sphere = []
    for w in words:
        s = [word_to_idx[w]]
        # radius 1 neighbors: change one coordinate to a different value
        for pos in range(n):
            cur = w[pos]
            for v in range(q):
                if v != cur:
                    w2 = list(w)
                    w2[pos] = v
                    s.append(word_to_idx[tuple(w2)])
        sphere.append(s)
    return words, sphere, total

def _verify(code_idx, sphere, total):
    # returns True iff every word index is covered by at least one codeword
    counts = [0] * total
    for ci in code_idx:
        for w in sphere[ci]:
            counts[w] += 1
    for c in counts:
        if c == 0:
            return False
    return True

def _coverage_counts(code_idx, sphere, total):
    counts = [0] * total
    for ci in code_idx:
        for w in sphere[ci]:
            counts[w] += 1
    return counts

def _max_flow_ford_fulkerson(source, sink, adj, cap):
    """
    Ford-Fulkerson with BFS (Edmonds-Karp) for max flow.
    adj: dict of dict, adj[u][v] = capacity
    cap: dict of dict, residual capacity
    Returns max flow value.
    """
    from collections import deque
    
    def bfs():
        # Find augmenting path from source to sink
        visited = {source}
        queue = deque([(source, [])])
        while queue:
            u, path = queue.popleft()
            if u == sink:
                return path
            for v in adj[u]:
                if v not in visited and cap[u][v] > 0:
                    visited.add(v)
                    queue.append((v, path + [(u, v)]))
        return None
    
    flow = 0
    while True:
        path = bfs()
        if path is None:
            break
        # Find bottleneck
        min_cap = min(cap[u][v] for u, v in path)
        flow += min_cap
        for u, v in path:
            cap[u][v] -= min_cap
            if v not in cap[u]:
                cap[u][v] = 0
            # Reverse edge
            if u not in cap.get(v, {}):
                cap[v][u] = 0
            cap[v][u] += min_cap
    return flow

def _min_vertex_cover_from_cut(source, sink, adj, cap, nodes):
    """
    After max flow, find min vertex cover in bipartite graph.
    For bipartite graph with left L and right R, min vertex cover = 
    (L \ S) ∪ (R ∩ S) where S is reachable from source in residual graph.
    """
    from collections import deque
    
    # Find nodes reachable from source in residual graph
    visited = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited and cap[u][v] > 0:
                visited.add(v)
                queue.append(v)
    
    # Determine min cut
    # In bipartite matching context:
    # Left side: space vertices (0 to total-1)
    # Right side: codeword candidates
    # Source -> space -> codeword -> sink
    
    # The min vertex cover in the bipartite graph corresponds to:
    # - Space vertices NOT in visited (on source side but not reachable means they're on sink side of cut)
    # Wait, let me think more carefully.
    
    # Actually, for set cover via flow, this is trickier. Let me use a different approach.
    # The min cut gives us which nodes are on the source side.
    # Nodes on source side that are space vertices: these are "covered" by the cut
    # Nodes on sink side that are codeword vertices: these are selected
    
    # Actually, let me reconsider. The standard reduction is:
    # Min set cover is NP-hard, so max flow gives an approximation or exact for special cases.
    
    # For our specific case with small numbers, let's use a different strategy:
    # Use the flow to identify which codewords are "essential"
    
    return visited

def _flow_based_reduction(code_idx, sphere, total, rng):
    """
    Try to reduce code size using coverage analysis.
    Identify codewords that can be removed or swapped.
    """
    code = list(code_idx)
    if len(code) < 2:
        return code
    
    counts = _coverage_counts(code, sphere, total)
    
    # Find codewords with minimal "unique coverage"
    # A codeword is "weak" if it covers few words uniquely
    unique_counts = {}
    for ci in code:
        unique = 0
        for w in sphere[ci]:
            if counts[w] == 1:
                unique += 1
        unique_counts[ci] = unique
    
    # Sort by unique coverage (ascending) - try to remove weakest first
    sorted_code = sorted(code, key=lambda x: unique_counts[x])
    
    # Try removing weakest codewords and repairing
    new_code = list(code)
    for ci in sorted_code:
        if ci not in new_code:
            continue
        
        # Try removing ci
        test_code = [c for c in new_code if c != ci]
        test_counts = _coverage_counts(test_code, sphere, total)
        uncovered = [w for w in range(total) if test_counts[w] == 0]
        
        if not uncovered:
            # Can remove without repair
            new_code = test_code
        elif len(uncovered) <= 12:
            # Try to find a replacement that covers all uncovered
            best_c = -1
            for c in range(total):
                if c in test_code:
                    continue
                covers_all = True
                for w in uncovered:
                    if w not in sphere[c]:
                        covers_all = False
                        break
                if covers_all:
                    best_c = c
                    break
            
            if best_c != -1:
                # Perfect replacement
                new_code = test_code + [best_c]
    
    # Local search on result
    return _local_search_fast(new_code, sphere, total)

def _local_search_fast(code_idx, sphere, total):
    """Fast local search: repeatedly delete redundant codewords, then try 2-exchanges."""
    code = sorted(set(code_idx))
    counts = [0] * total
    for ci in code:
        for w in sphere[ci]:
            counts[w] += 1
    
    # Phase 1: Delete redundant codewords
    changed = True
    while changed:
        changed = False
        new_code = []
        for ci in code:
            redundant = True
            for w in sphere[ci]:
                if counts[w] <= 1:
                    redundant = False
                    break
            if redundant:
                for w in sphere[ci]:
                    counts[w] -= 1
                changed = True
            else:
                new_code.append(ci)
        code = new_code
    
    # Phase 2: Try 2-exchanges (remove 2, add 1 that covers all lost coverage)
    n = len(code)
    if n < 3:
        return code
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                # Try removing code[i] and code[j]
                remaining = [c for k, c in enumerate(code) if k != i and k != j]
                rem_counts = [0] * total
                for ci in remaining:
                    for w in sphere[ci]:
                        rem_counts[w] += 1
                
                # Find uncovered words
                uncovered = set()
                for w in sphere[code[i]]:
                    if rem_counts[w] == 0:
                        uncovered.add(w)
                for w in sphere[code[j]]:
                    if rem_counts[w] == 0:
                        uncovered.add(w)
                
                if not uncovered:
                    # Can remove both
                    code = remaining
                    improved = True
                    break
                elif len(uncovered) <= 12:  # Small enough to try to cover with one codeword
                    # Try to find a single codeword that covers all uncovered
                    best_c = -1
                    for c in range(total):
                        if c in remaining:
                            continue
                        covers_all = True
                        for w in uncovered:
                            if w not in sphere[c]:
                                covers_all = False
                                break
                        if covers_all:
                            best_c = c
                            break
                    
                    if best_c != -1:
                        code = remaining + [best_c]
                        improved = True
                        break
            
            if improved:
                break
        
        if improved:
            # Recompute counts after modification
            counts = [0] * total
            for ci in code:
                for w in sphere[ci]:
                    counts[w] += 1
            n = len(code)
    
    return code

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 110.0  # safety margin under 120/150s limits

    words, sphere, total = _build_space(q, n, radius)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Seed: tetracode direct sum -> 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_idx = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            seed_idx.append(word_to_idx[tuple(w)])
    best_code = sorted(set(seed_idx))
    best_size = len(best_code)

    def local_search(code_idx):
        # Repeatedly delete any redundant codeword until fixed point.
        counts = _coverage_counts(list(code_idx), sphere, total)
        code = sorted(set(code_idx))
        changed = True
        while changed:
            changed = False
            new_code = []
            for ci in code:
                # check redundancy: after removing ci, every word in sphere[ci]
                # must still have count > 1 (i.e., covered by some other)
                redundant = True
                for w in sphere[ci]:
                    if counts[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    for w in sphere[ci]:
                        counts[w] -= 1
                    changed = True
                else:
                    new_code.append(ci)
            code = new_code
        return code

    def greedy_from_uncovered(uncovered_set):
        # uncovered_set: set of word indices currently uncovered
        # returns list of centers that cover all of them (additions)
        chosen = []
        uncov = set(uncovered_set)
        while uncov:
            best_c = -1
            best_cnt = -1
            for c in range(total):
                cnt = 0
                for w in sphere[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_cnt <= 0:
                break  # should not happen
            chosen.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
        return chosen

    def repair_and_local_search(code_idx, uncovered):
        # add centers to cover uncovered, then local-search
        added = greedy_from_uncovered(uncovered)
        combined = list(code_idx) + added
        return local_search(combined)

    def swap_local_search(code_idx, rng, max_swaps=30):
        # Try to swap codewords to enable further deletions
        code = list(code_idx)
        code_set = set(code)
        counts = _coverage_counts(code, sphere, total)
        
        improved = True
        swaps_made = 0
        while improved and swaps_made < max_swaps:
            improved = False
            if not code:
                break
            
            # Try each codeword as potential removal candidate
            for out_idx in range(len(code)):
                c_out = code[out_idx]
                
                # Find words covered ONLY by c_out
                critical_words = []
                for w in sphere[c_out]:
                    if counts[w] == 1:
                        critical_words.append(w)
                
                if not critical_words:
                    # Can remove without replacement
                    new_code = [c for i, c in enumerate(code) if i != out_idx]
                    new_code = local_search(new_code)
                    if len(new_code) < len(code):
                        code = new_code
                        code_set = set(code)
                        counts = _coverage_counts(code, sphere, total)
                        improved = True
                        swaps_made += 1
                        break
                    continue

                # Find a candidate c_in that covers all critical_words
                best_c_in = -1
                for c_in in range(total):
                    if c_in in code_set:
                        continue
                    covers_all = True
                    for w in critical_words:
                        if w not in sphere[c_in]:
                            covers_all = False
                            break
                    if covers_all:
                        best_c_in = c_in
                        break
                
                if best_c_in != -1:
                    code[out_idx] = best_c_in
                    code_set.discard(c_out)
                    code_set.add(best_c_in)
                    counts = _coverage_counts(code, sphere, total)
                    new_code = local_search(code)
                    if len(new_code) < len(code):
                        code = new_code
                        code_set = set(code)
                        counts = _coverage_counts(code, sphere, total)
                        improved = True
                        swaps_made += 1
                        break
        return code

    def flow_refinement(code_idx, rng):
        """Use coverage analysis to identify removable codewords."""
        code = list(code_idx)
        if len(code) < 2:
            return code
        
        # Identify codewords with minimal unique coverage
        counts = _coverage_counts(code, sphere, total)
        unique_counts = {}
        for ci in code:
            unique = 0
            for w in sphere[ci]:
                if counts[w] == 1:
                    unique += 1
            unique_counts[ci] = unique
        
        # Sort by unique coverage (ascending)
        sorted_code = sorted(code, key=lambda x: unique_counts[x])
        
        # Try removing weakest codewords
        new_code = list(code)
        for ci in sorted_code[:10]:  # Try top 10 weakest
            if ci not in new_code:
                continue
            
            test_code = [c for c in new_code if c != ci]
            test_counts = _coverage_counts(test_code, sphere, total)
            uncovered = [w for w in range(total) if test_counts[w] == 0]
            
            if not uncovered:
                new_code = test_code
            elif len(uncovered) <= 12:
                # Try to find replacement that covers all uncovered
                best_c = -1
                for c in range(total):
                    if c in test_code:
                        continue
                    covers_all = True
                    for w in uncovered:
                        if w not in sphere[c]:
                            covers_all = False
                            break
                    if covers_all:
                        best_c = c
                        break
                
                if best_c != -1:
                    new_code = test_code + [best_c]
        
        return _local_search_fast(new_code, sphere, total)

    rng = random.Random(20240517)

    # Pass 1: local search on the 81-word seed
    cand = local_search(best_code)
    if len(cand) < best_size:
        best_size = len(cand)
        best_code = cand

    # Pass 2: flow-based refinement
    if time.time() < DEADLINE - 40:
        flow_cand = flow_refinement(best_code, rng)
        if len(flow_cand) < best_size:
            best_size = len(flow_cand)
            best_code = flow_cand

    # Pass 3: full greedy from scratch (random tie-breaks) + local search, multiple times
    for trial in range(60):
        if time.time() > DEADLINE - 35:
            break
        centers = list(range(total))
        rng.shuffle(centers)
        uncov = set(range(total))
        code = []
        while uncov and len(code) < 85:
            best_c = -1
            best_cnt = -1
            for c in centers:
                if not uncov:
                    break
                cnt = 0
                for w in sphere[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt <= 0:
                break
            code.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
            centers = [c for c in centers if c != best_c]
            rng.shuffle(centers)
        if uncov:
            continue
        code2 = local_search(code)
        code2_flow = flow_refinement(code2, rng)
        code2_swapped = swap_local_search(code2_flow, rng)
        if len(code2_swapped) < best_size:
            best_size = len(code2_swapped)
            best_code = code2_swapped
        elif len(code2_flow) < best_size:
            best_size = len(code2_flow)
            best_code = code2_flow
        elif len(code2) < best_size:
            best_size = len(code2)
            best_code = code2

    # Pass 4: perturbation from best_code
    iterations = 0
    while time.time() < DEADLINE - 15:
        iterations += 1
        code = list(best_code)
        k = len(code)
        if k < 2:
            break
        
        if iterations % 3 == 0:
            # Flow-based perturbation
            flow_cand = flow_refinement(code, rng)
            flow_swapped = swap_local_search(flow_cand, rng)
            if len(flow_swapped) < best_size:
                best_size = len(flow_swapped)
                best_code = flow_swapped
            elif len(flow_cand) < best_size:
                best_size = len(flow_cand)
                best_code = flow_cand
        elif iterations % 2 == 0:
            # Drop and Repair - drop 2 random codewords
            if k >= 3:
                drop_idx1 = rng.randrange(k)
                drop_idx2 = rng.randrange(k - 1)
                if drop_idx2 >= drop_idx1:
                    drop_idx2 += 1
                new_code = [c for j, c in enumerate(code) if j != drop_idx1 and j != drop_idx2]
            else:
                drop_idx = rng.randrange(k)
                new_code = [c for j, c in enumerate(code) if j != drop_idx]
            counts = _coverage_counts(new_code, sphere, total)
            uncovered = set()
            for w in range(total):
                if counts[w] == 0:
                    uncovered.add(w)
            repaired = repair_and_local_search(new_code, uncovered)
            repaired_flow = flow_refinement(repaired, rng)
            repaired_swapped = swap_local_search(repaired_flow, rng)
            if len(repaired_swapped) < best_size:
                best_size = len(repaired_swapped)
                best_code = repaired_swapped
            elif len(repaired_flow) < best_size:
                best_size = len(repaired_flow)
                best_code = repaired_flow
            elif len(repaired) < best_size:
                best_size = len(repaired)
                best_code = repaired
        else:
            # Swap Perturbation
            out_idx = rng.randrange(k)
            c_out = code[out_idx]
            c_in = rng.randrange(total)
            while c_in in code:
                c_in = rng.randrange(total)
            
            new_code = list(code)
            new_code[out_idx] = c_in
            
            if _verify(new_code, sphere, total):
                reduced = local_search(new_code)
                reduced_flow = flow_refinement(reduced, rng)
                reduced_swapped = swap_local_search(reduced_flow, rng)
                if len(reduced_swapped) < best_size:
                    best_size = len(reduced_swapped)
                    best_code = reduced_swapped
                elif len(reduced_flow) < best_size:
                    best_size = len(reduced_flow)
                    best_code = reduced_flow
                elif len(reduced) < best_size:
                    best_size = len(reduced)
                    best_code = reduced
            else:
                counts = _coverage_counts(new_code, sphere, total)
                uncovered = set()
                for w in range(total):
                    if counts[w] == 0:
                        uncovered.add(w)
                if uncovered:
                    repaired = repair_and_local_search(new_code, uncovered)
                    repaired_flow = flow_refinement(repaired, rng)
                    repaired_swapped = swap_local_search(repaired_flow, rng)
                    if len(repaired_swapped) < best_size:
                        best_size = len(repaired_swapped)
                        best_code = repaired_swapped
                    elif len(repaired_flow) < best_size:
                        best_size = len(repaired_flow)
                        best_code = repaired_flow
                    elif len(repaired) < best_size:
                        best_size = len(repaired)
                        best_code = repaired

    if not _verify(best_code, sphere, total):
        best_code = sorted(set(seed_idx))

    return [list(words[i]) for i in sorted(set(best_code))]
