import time
import random
import sys

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    """
    triples = []
    seen = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Store sorted tuple to identify the set of 3 elements
                    # The constraint is on the set {a,b,c}
                    key = tuple(sorted((a, b, c)))
                    if key not in seen:
                        seen.add(key)
                        triples.append(key)
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    N = 180
    
    triples = _precompute_triples(N)
    T = len(triples)
    
    # Map element to list of triple indices
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    
    # Precompute degree
    degree = {i: len(elem_triples[i]) for i in range(1, N + 1)}
    
    best_set = set()
    best_size = 0
    
    def greedy_add(order):
        """
        Greedily add elements in the given order.
        Returns the set and the counts array.
        """
        S = set()
        counts = [0] * T
        
        for x in order:
            # Check if adding x completes any triple
            can_add = True
            for t_idx in elem_triples[x]:
                if counts[t_idx] == 2:
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
                for t_idx in elem_triples[x]:
                    counts[t_idx] += 1
        
        return S, counts

    def local_search_add_back(S, counts, random_seed):
        """
        Try to add back elements not in S.
        """
        random.seed(random_seed)
        S = set(S)
        counts = list(counts)
        
        # List of elements not in S
        removed = [x for x in range(1, N + 1) if x not in S]
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            
            # Shuffle to break symmetry
            random.shuffle(removed)
            
            for x in removed:
                if x in S:
                    continue
                can_add = True
                for t_idx in elem_triples[x]:
                    if counts[t_idx] == 2:
                        can_add = False
                        break
                
                if can_add:
                    S.add(x)
                    removed.remove(x)
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
                    improved = True
        
        return S, counts

    def local_search_swap(S, counts, random_seed):
        """
        Try to improve by removing one element and adding others.
        """
        random.seed(random_seed)
        S = set(S)
        counts = list(counts)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            
            # Try removing each element in S
            S_list = list(S)
            random.shuffle(S_list)
            
            for x in S_list:
                if x not in S:
                    continue
                
                # Temporarily remove x
                S.remove(x)
                for t_idx in elem_triples[x]:
                    counts[t_idx] -= 1
                
                # Try to add as many elements as possible from the complement
                added = []
                # We need to efficiently find addable elements.
                # Iterate through all elements not in S
                for y in range(1, N + 1):
                    if y in S:
                        continue
                    can_add = True
                    for t_idx in elem_triples[y]:
                        if counts[t_idx] == 2:
                            can_add = False
                            break
                    if can_add:
                        S.add(y)
                        added.append(y)
                        for t_idx in elem_triples[y]:
                            counts[t_idx] += 1
                
                # Check if we gained size
                if len(added) > 1:
                    # Net gain: removed 1, added > 1
                    improved = True
                    # Keep the new state
                else:
                    # Revert
                    for y in added:
                        S.remove(y)
                        for t_idx in elem_triples[y]:
                            counts[t_idx] -= 1
                    S.add(x)
                    for t_idx in elem_triples[x]:
                        counts[t_idx] += 1
        
        return S, counts

    # Main loop with restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Create a random order, biased towards low degree elements
        # We can use a weighted random selection or just sort by (degree + random_noise)
        noise = [random.random() * 10 for _ in range(N + 1)]
        # Sort key: degree + noise
        order = list(range(1, N + 1))
        order.sort(key=lambda x: degree[x] + noise[x])
        
        S_greedy, counts_greedy = greedy_add(order)
        
        # Local search
        S_ls, counts_ls = local_search_add_back(S_greedy, counts_greedy, i)
        S_final, counts_final = local_search_swap(S_ls, counts_ls, i + 1000)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = set(S_final)
            
            # Optional: If we hit a very high size, we can spend more time polishing
            # but for now, just keep going.

    # Final polish on the best set found
    if best_set and time.time() < deadline - 3.0:
        # Re-compute counts for best_set
        counts = [0] * T
        for x in best_set:
            for t_idx in elem_triples[x]:
                counts[t_idx] += 1
        
        # One more pass of add-back and swap
        S_polished, counts_polished = local_search_add_back(best_set, counts, 9999)
        S_polished, counts_polished = local_search_swap(S_polished, counts_polished, 8888)
        
        if len(S_polished) > best_size:
            best_set = S_polished
            best_size = len(best_set)

    # Fallback
    if not best_set:
        best_set = set(range(90, 181))
        
    # Validity check
    valid = True
    for (a, b, c) in triples:
        if a in best_set and b in best_set and c in best_set:
            valid = False
            break
            
    if not valid:
        # Remove one element from each violating triple
        for (a, b, c) in triples:
            if a in best_set and b in best_set and c in best_set:
                best_set.remove(c)

    return list(best_set)

# EVOLVE-BLOCK-END
