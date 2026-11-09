import time
import random
import itertools
import sys

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Uses bitmask-based greedy set cover and iterated local search to minimize size.
    """
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s, we aim to finish by 145s to be safe.
    deadline = start_time + 140.0
    
    # Handle the specific task instance K_3(7,1)
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** 7  # 2187
        
        # Generate all words and map to indices
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbor lists as Python lists
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    neighbors[i].append(idx)
        
        # Precompute neighbor masks as Python integers (bitsets)
        # Note: Python integers have arbitrary precision, so 1 << 2187 is fine.
        neighbor_masks = [0] * N
        for i in range(N):
            mask = 0
            for nb in neighbors[i]:
                mask |= (1 << nb)
            neighbor_masks[i] = mask
        
        full_mask = (1 << N) - 1
        
        def indices_to_words(indices):
            return [list(all_words[idx]) for idx in indices]
        
        def greedy_set_cover(seed=None):
            """Randomized greedy set cover using bitmasks."""
            if seed is not None:
                random.seed(seed)
            
            covered = 0
            code_indices = []
            candidates = list(range(N))
            random.shuffle(candidates)
            
            while covered != full_mask:
                best_gain = -1
                best_idx = -1
                
                for idx in candidates:
                    gain = (neighbor_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        # Early termination if we found a perfect cover (unlikely in greedy)
                        if gain == (full_mask ^ covered).bit_count():
                            break
                
                if best_gain == 0:
                    break
                
                code_indices.append(best_idx)
                covered |= neighbor_masks[best_idx]
            
            return code_indices
        
        def optimize(initial_indices, time_limit=None):
            """
            Local search: Removal passes + Swap moves.
            Returns a list of indices.
            """
            if time_limit is None:
                time_limit = deadline - 2.0
            
            current_indices = list(initial_indices)
            random.shuffle(current_indices)
            
            cover_count = [0] * N
            in_code = [False] * N
            
            for idx in current_indices:
                in_code[idx] = True
                for nb in neighbors[idx]:
                    cover_count[nb] += 1
            
            improved = True
            no_improve_count = 0
            max_no_improve = 50
            
            while improved and time.time() < time_limit:
                improved = False
                random.shuffle(current_indices)
                
                # 1. Removal Pass
                i = 0
                while i < len(current_indices):
                    if time.time() > time_limit:
                        break
                    
                    idx = current_indices[i]
                    redundant = True
                    for nb in neighbors[idx]:
                        if cover_count[nb] <= 1:
                            redundant = False
                            break
                    
                    if redundant:
                        for nb in neighbors[idx]:
                            cover_count[nb] -= 1
                        in_code[idx] = False
                        current_indices.pop(i)
                        improved = True
                        no_improve_count = 0
                    else:
                        i += 1
                
                # 2. Swap Pass
                # Try to swap A (in code) with B (not in code) if it allows future removals
                # or simply maintains coverage while potentially leading to a better configuration.
                # A valid swap A->B requires that every word covered ONLY by A is also covered by B.
                # i.e., {x | cover_count[x] == 1 and x in N(A)} is a subset of N(B).
                
                # To speed up, we only attempt swaps if we are stuck or to explore.
                # We limit swap attempts per iteration to avoid O(N^2) explosion.
                swap_attempts = 200
                for _ in range(swap_attempts):
                    if time.time() > time_limit:
                        break
                    if len(current_indices) < 180: # Don't try to reduce below known bounds aggressively if already small
                        break
                        
                    idx_a = random.choice(current_indices)
                    
                    # Pick random B not in current set
                    # Optimization: Instead of random sampling until unique, we can precompute
                    # a list of non-code indices, but it's expensive to maintain.
                    # For N=2187, random sampling is fast enough.
                    idx_b = random.randint(0, N - 1)
                    while in_code[idx_b]:
                        idx_b = random.randint(0, N - 1)
                    
                    mask_a = neighbor_masks[idx_a]
                    mask_b = neighbor_masks[idx_b]
                    
                    # Check validity: 
                    # Words covered by A but not by B must have cover_count > 1 (so they remain covered by others).
                    # Equivalently: (N(A) \ N(B)) must not contain any word with cover_count == 1.
                    # Or: For all x in N(A), if cover_count[x] == 1, then x must be in N(B).
                    
                    valid_swap = True
                    for nb in neighbors[idx_a]:
                        if cover_count[nb] == 1:
                            # This word is uniquely covered by A.
                            # It MUST be covered by B.
                            if not ((mask_b >> nb) & 1):
                                valid_swap = False
                                break
                    
                    if valid_swap:
                        # Perform swap: remove A, add B
                        for nb in neighbors[idx_a]:
                            cover_count[nb] -= 1
                        in_code[idx_a] = False
                        
                        for nb in neighbors[idx_b]:
                            cover_count[nb] += 1
                        in_code[idx_b] = True
                        
                        # Replace A with B in current_indices
                        # Find index of A
                        pos_a = current_indices.index(idx_a)
                        current_indices[pos_a] = idx_b
                        improved = True
                        no_improve_count = 0
                        
                        # After a swap, we might have created redundancies.
                        # Do a quick removal pass immediately.
                        i = 0
                        while i < len(current_indices):
                            if time.time() > time_limit:
                                break
                            idx = current_indices[i]
                            redundant = True
                            for nb in neighbors[idx]:
                                if cover_count[nb] <= 1:
                                    redundant = False
                                    break
                            if redundant:
                                for nb in neighbors[idx]:
                                    cover_count[nb] -= 1
                                in_code[idx] = False
                                current_indices.pop(i)
                                improved = True
                                no_improve_count = 0
                            else:
                                i += 1
            
            return current_indices

        def iterated_local_search(seed=None):
            """
            ILS: Start with greedy, optimize, then kick (perturb) and re-optimize.
            """
            if seed is not None:
                random.seed(seed)
            
            # Initial greedy cover
            current_code = greedy_set_cover()
            
            # Optimize
            current_code = optimize(current_code, deadline - 10)
            
            best_code = list(current_code)
            best_size = len(best_code)
            
            while time.time() < deadline - 5:
                # Kick: Randomly remove k codewords and re-add greedily
                # k is small, e.g., 5-10
                kick_size = random.randint(3, 8)
                
                # Select random codewords to remove
                if len(current_code) > kick_size:
                    remove_indices = set(random.sample(current_code, kick_size))
                else:
                    remove_indices = set(current_code)
                
                new_code = [c for c in current_code if c not in remove_indices]
                
                # Re-add greedily to ensure coverage
                covered = 0
                for idx in new_code:
                    covered |= neighbor_masks[idx]
                
                # Greedily add back until full coverage
                # We prefer adding words that were in the original best_code or close to it?
                # For simplicity, standard greedy on remaining uncovered.
                candidates = list(set(range(N)) - set(new_code))
                random.shuffle(candidates)
                
                while covered != full_mask:
                    best_gain = -1
                    best_idx = -1
                    for idx in candidates:
                        gain = (neighbor_masks[idx] & ~covered).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx
                            if gain == (full_mask ^ covered).bit_count():
                                break
                    if best_gain == 0:
                        break
                    new_code.append(best_idx)
                    covered |= neighbor_masks[best_idx]
                
                # Optimize the new code
                new_code = optimize(new_code, deadline - 5)
                
                if len(new_code) < best_size:
                    best_size = len(new_code)
                    best_code = list(new_code)
                    current_code = list(new_code)
                else:
                    # With some probability, accept a worse solution to escape local minima (Simulated Annealing style)
                    # But for this problem, we usually want to keep the best so far.
                    # However, ILS typically keeps the best.
                    # Let's stick to keeping the best to avoid losing progress.
                    pass
                
            return best_code

        # Run ILS
        best_set = iterated_local_search(seed=42)
        
        # Final verification and cleanup
        # Ensure no duplicates (shouldn't happen with sets, but good practice)
        best_set = list(set(best_set))
        
        # One last optimization pass on the best found
        final_set = optimize(best_set, deadline - 1)
        
        return indices_to_words(final_set)
    
    # Generic fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nbs = [word_to_idx[w]]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nbs.append(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(len(all_words)))
    code_set = []
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(len(all_words)):
            if idx in code_set:
                continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1:
            break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]

if __name__ == "__main__":
    # Quick test
    res = construct(3, 7, 1)
    print(f"Found code size: {len(res)}")
