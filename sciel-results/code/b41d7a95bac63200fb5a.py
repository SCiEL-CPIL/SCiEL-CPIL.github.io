import itertools
import random
import time

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations and sort them lexicographically
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from prefix to list of indices of permutations starting with that prefix
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
    
    def run_greedy(seed):
        rng = random.Random(seed)
        visited = [False] * N
        visited[0] = True
        current_word = list(perms[0])
        
        for _ in range(N - 1):
            found = False
            # Try max overlap first
            for k in range(n - 1, 0, -1):
                suffix = tuple(current_word[-k:])
                candidates = prefix_to_indices.get(suffix, [])
                
                if not candidates:
                    continue
                
                # Filter unvisited candidates
                unvisited = [idx for idx in candidates if not visited[idx]]
                
                if not unvisited:
                    continue
                
                # Pick a random candidate
                best_idx = rng.choice(unvisited)
                visited[best_idx] = True
                next_perm = perms[best_idx]
                # Append the new symbols
                current_word.extend(next_perm[k:])
                found = True
                break
            
            if not found:
                # Stuck, this run failed
                return None
                
        return current_word

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_word = None
    best_len = float('inf')
    seed = 0
    
    while time.time() < deadline:
        word = run_greedy(seed)
        if word is not None:
            l = len(word)
            if l < best_len:
                best_len = l
                best_word = word
        seed += 1
        
        # If we found a very short word, we might stop early?
        # Theoretical minimum for n=7 is 5905.
        # If we hit that, we can stop.
        if n == 7 and best_len <= 5905:
            break
        if n == 6 and best_len <= 872: # Known shortest for 6 is 872
            break

    if best_word is None:
        # Fallback: return a trivial valid superpermutation if greedy failed completely
        # This should not happen if we have enough time, but better safe than sorry.
        # Trivial: just concatenate all perms
        return "".join(map(str, [x for p in perms for x in p]))
    
    return "".join(map(str, best_word))
