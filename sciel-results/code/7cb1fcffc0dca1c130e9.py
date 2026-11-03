import time
import random
import math
from heapq import heappush, heappop

def _precompute_neighbors(q, n, radius):
    total = q ** n
    # For radius 1, neighbors are the word itself and words differing in 1 position
    neigh = [[] for _ in range(total)]
    neigh_sets = [set() for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_sets[idx].add(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
                    neigh_sets[idx].add(new_idx)
    return neigh, neigh_sets, total

def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_sets, _ = _precompute_neighbors(q, n, radius)
    
    # Helper to check if a set of codewords covers all space
    def is_valid_code(code_list):
        covered = [False] * total
        for c in code_list:
            for u in neigh[c]:
                covered[u] = True
        # Check if all covered
        for u in range(total):
            if not covered[u]:
                return False
        return True

    # Greedy set cover initialization
    def greedy_init(seed):
        rng_init = random.Random(seed)
        counts = [0] * total
        code = []
        remaining = [True] * total
        rem_count = total
        
        def add_to_code(c):
            nonlocal rem_count
            code.append(c)
            for u in neigh[c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1

        def remove_from_code(c):
            nonlocal rem_count
            code.remove(c)
            for u in neigh[c]:
                counts[u] -= 1
                if counts[u] == 0:
                    remaining[u] = True
                    rem_count += 1

        # Greedy selection
        while rem_count > 0:
            best_c = -1
            best_gain = -1
            for c in range(total):
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            add_to_code(best_c)
            
        return code, counts

    # Aggressive local search using priority queue for removal
    def local_search(code, counts, deadline_local):
        # Make a copy
        code = code[:]
        counts = counts[:]
        
        # We want to remove any codeword that is redundant.
        # A codeword c is redundant if for all u in neigh[c], counts[u] > 1.
        # We can iterate until no more removals are possible.
        
        changed = True
        while changed and time.monotonic() < deadline_local:
            changed = False
            # Iterate in random order to break symmetries
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                c = code[idx]
                # Check if c is removable
                removable = True
                for u in neigh[c]:
                    if counts[u] == 1:
                        removable = False
                        break
                if removable:
                    # Remove c
                    for u in neigh[c]:
                        counts[u] -= 1
                    code.pop(idx)
                    changed = True
        return code, counts

    # Iterated Local Search
    best_code = None
    best_size = float('inf')
    
    # Multi-start ILS
    num_starts = 20
    start_times = []
    for start in range(num_starts):
        if time.monotonic() > deadline:
            break
            
        # Greedy initialization with different seeds
        code, counts = greedy_init(42 + start)
        
        # Local search
        code, counts = local_search(code, counts, deadline)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        # If we hit the lower bound (71) or known optimum (73), we can stop early?
        # Lower bound is 71, best known is 73.
        if best_size <= 73:
            # We found a very good solution. Maybe try a few more to see if we can get 71/72?
            # But 73 is likely the target.
            pass

        # Perturbation: k-swap
        if time.monotonic() > deadline:
            break
            
        k = 2 # Try swapping 2 codewords
        if len(code) > k:
            # Pick k random codewords to remove
            remove_indices = rng.sample(range(len(code)), k)
            removed = [code[i] for i in sorted(remove_indices, reverse=True)]
            for i in sorted(remove_indices, reverse=True):
                c = code[i]
                for u in neigh[c]:
                    counts[u] -= 1
                code.pop(i)
                
            # Try to add k new codewords to restore validity
            # We need to add codewords that cover the uncovered points.
            # Find uncovered points
            uncovered = []
            for u in range(total):
                if counts[u] == 0:
                    uncovered.append(u)
            
            # If no uncovered points, we just removed redundant codewords?
            # No, we removed k random ones. If it's still valid, great.
            # If not, we need to add.
            if not uncovered:
                # Valid, maybe we can even remove more?
                code, counts = local_search(code, counts, deadline)
            else:
                # Add k random codewords that cover uncovered points
                # Greedily add
                for _ in range(k):
                    if time.monotonic() > deadline:
                        break
                    # Find codeword covering most uncovered
                    best_c = -1
                    best_gain = -1
                    for c in range(total):
                        if c in code:
                            continue
                        gain = 0
                        for u in neigh[c]:
                            if counts[u] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                    if best_c == -1 or best_gain == 0:
                        # Random add
                        best_c = rng.randint(0, total - 1)
                        if best_c in code:
                            continue
                    code.append(best_c)
                    for u in neigh[best_c]:
                        counts[u] += 1
                    
                # Check validity
                if is_valid_code(code):
                    code, counts = local_search(code, counts, deadline)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                else:
                    # Restore original? Or just discard this iteration.
                    # Discarding is safer.
                    pass

    # If we didn't find anything? (Should not happen)
    if best_code is None:
        code, counts = greedy_init(42)
        code, counts = local_search(code, counts, deadline)
        best_code = code

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
