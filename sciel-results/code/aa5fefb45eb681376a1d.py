"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # For the specific case q=3, n=6, radius=1, use optimized greedy set cover
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(n)]

        # Precompute for each word, the 13 words within Hamming distance 1
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        def greedy_set_cover(seed_val):
            """Greedy set cover with a given random seed for tie-breaking."""
            rng = random.Random(seed_val)
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []
            
            while uncovered.sum() > 0:
                if time.time() > deadline:
                    break
                counts = np.sum(uncovered[neighbors], axis=1)
                max_count = counts.max()
                if max_count == 0:
                    break
                # Among candidates with max count, pick randomly for diversity
                best_indices = np.where(counts == max_count)[0]
                best_candidate = best_indices[rng.randint(0, len(best_indices) - 1)]
                codewords.append(int(best_candidate))
                uncovered[neighbors[best_candidate]] = 0
            
            if uncovered.sum() > 0:
                return None
            return codewords

        def try_remove_redundant(codewords):
            """Remove any codeword whose coverage is fully redundant."""
            improved = True
            while improved:
                if time.time() > deadline - 5:
                    break
                improved = False
                coverage_counts = np.zeros(total, dtype=np.int32)
                for idx_c in codewords:
                    coverage_counts[neighbors[idx_c]] += 1
                for i, idx_c in enumerate(codewords):
                    if np.min(coverage_counts[neighbors[idx_c]]) >= 2:
                        codewords.pop(i)
                        improved = True
                        break
            return codewords

        def simulated_annealing(codewords, max_time):
            """Simulated annealing to reduce codeword count via swaps."""
            if time.time() > deadline - 5:
                return codewords
            
            best_codewords = list(codewords)
            current_codewords = list(codewords)
            current_len = len(current_codewords)
            best_len = current_len
            
            # Build coverage structure
            def build_coverage(cw):
                cov = np.zeros(total, dtype=np.int32)
                for idx_c in cw:
                    cov[neighbors[idx_c]] += 1
                return cov
            
            current_cov = build_coverage(current_codewords)
            in_set = np.zeros(total, dtype=np.int8)
            for idx_c in current_codewords:
                in_set[idx_c] = 1
            
            T = 1.0
            cooling = 0.95
            no_improve_count = 0
            
            while time.time() < max_time:
                # Pick a random codeword to potentially swap
                i = random.randint(0, len(current_codewords) - 1)
                idx_c = current_codewords[i]
                
                # Find words only covered by this codeword
                only_covered = neighbors[idx_c][current_cov[neighbors[idx_c]] == 1]
                
                if len(only_covered) == 0:
                    # Can remove directly
                    current_codewords.pop(i)
                    current_cov[neighbors[idx_c]] -= 1
                    in_set[idx_c] = 0
                    current_len -= 1
                    if current_len < best_len:
                        best_len = current_len
                        best_codewords = list(current_codewords)
                    no_improve_count = 0
                    T = max(0.1, T * cooling)
                    continue
                
                # Try to find a replacement that covers all only_covered words
                # Candidates: words that are neighbors of at least one only_covered word
                candidates = set()
                for wc in only_covered:
                    for nb in neighbors[wc]:
                        if not in_set[nb]:
                            candidates.add(int(nb))
                
                if not candidates:
                    T = max(0.1, T * cooling)
                    continue
                
                # Pick a random candidate
                cand_list = list(candidates)
                new_idx = cand_list[random.randint(0, len(cand_list) - 1)]
                
                # Check if replacing idx_c with new_idx maintains coverage
                # Words that would lose coverage: those covered by idx_c but not by new_idx
                lost = neighbors[idx_c][current_cov[neighbors[idx_c]] == 1]
                # After removing idx_c, these words need to be covered by new_idx
                # new_idx covers neighbors[new_idx]
                # Check: are all 'lost' words in neighbors[new_idx]?
                new_covers = set(neighbors[new_idx].tolist())
                lost_set = set(lost.tolist())
                
                if lost_set.issubset(new_covers):
                    # Valid swap
                    current_codewords[i] = new_idx
                    current_cov[neighbors[idx_c]] -= 1
                    current_cov[neighbors[new_idx]] += 1
                    in_set[idx_c] = 0
                    in_set[new_idx] = 1
                    no_improve_count = 0
                else:
                    # Invalid swap - accept with probability based on temperature
                    # Penalty: number of uncovered words
                    uncovered_after = len(lost_set - new_covers)
                    prob = np.exp(-uncovered_after / max(T, 0.01))
                    if random.random() < prob:
                        # Accept invalid move (will need to fix later)
                        current_codewords[i] = new_idx
                        current_cov[neighbors[idx_c]] -= 1
                        current_cov[neighbors[new_idx]] += 1
                        in_set[idx_c] = 0
                        in_set[new_idx] = 1
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                
                T = max(0.05, T * cooling)
                
                if no_improve_count > 50:
                    no_improve_count = 0
                    # Restart temperature
                    T = 0.5
            
            return best_codewords

        def add_then_remove(codewords):
            """Try adding a random codeword then removing redundant ones."""
            if time.time() > deadline - 5:
                return codewords
            
            best = list(codewords)
            best_len = len(best)
            
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                if len(codewords) >= best_len:
                    break
                
                # Add a random non-codeword
                in_set = np.zeros(total, dtype=np.int8)
                for idx_c in codewords:
                    in_set[idx_c] = 1
                non_cw = np.where(in_set == 0)[0]
                if len(non_cw) == 0:
                    break
                new_idx = int(non_cw[random.randint(0, len(non_cw) - 1)])
                codewords.append(new_idx)
                
                # Try to remove redundant
                codewords = try_remove_redundant(codewords)
                
                if len(codewords) < best_len:
                    best_len = len(codewords)
                    best = list(codewords)
            
            return best

        # Main optimization loop
        best_result = None
        best_len = float('inf')
        
        # Try multiple random greedy initializations
        num_initializations = 10
        time_per_init = (deadline - start_time - 20) / max(num_initializations, 1)
        
        for seed_val in range(num_initializations):
            if time.time() > deadline - 20:
                break
            
            init_deadline = start_time + (deadline - start_time) * (seed_val + 1) / num_initializations - 5
            codewords = greedy_set_cover(seed_val)
            
            if codewords is None:
                continue
            
            # Remove redundant
            codewords = try_remove_redundant(codewords)
            
            # Simulated annealing
            sa_deadline = min(init_deadline, time.time() + max(1, (deadline - start_time) / (num_initializations * 2)))
            codewords = simulated_annealing(codewords, sa_deadline)
            
            # Add-then-remove
            codewords = add_then_remove(codewords)
            
            # Final redundant removal
            codewords = try_remove_redundant(codewords)
            
            if len(codewords) < best_len:
                best_len = len(codewords)
                best_result = list(codewords)
            
            if best_len <= 73:
                break
        
        # If we found something good, use it; otherwise fallback
        if best_result is not None and best_len < 81:
            final_codewords = []
            for idx_c in best_result:
                final_codewords.append([(idx_c // powers[p]) % 3 for p in range(n)])
            return final_codewords
        
        # Fallback to direct sum
        return get_direct_sum()

    # General case: fallback to direct sum
    return get_direct_sum()
# EVOLVE-BLOCK-END
