"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17

        # Precompute powers of 3 for fast index manipulation
        pow3 = [3 ** i for i in range(n + 1)]

        def int_to_word(idx):
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))

        def word_to_int(w):
            result = 0
            for d in w:
                result = result * 3 + d
            return result

        # Build sphere index matrix: shape (6561, 17)
        sphere_idx_2d = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            w = int_to_word(idx)
            sphere_idx_2d[idx, 0] = idx
            k = 1
            for pos in range(n):
                orig = w[pos]
                base = pow3[n - 1 - pos]
                for d in range(q):
                    if d != orig:
                        sphere_idx_2d[idx, k] = idx + (d - orig) * base
                        k += 1

        # Phase 1: Fast greedy set cover (vectorized)
        uncovered = np.ones(total_space, dtype=bool)
        uncovered_count = total_space
        code_indices = []

        greedy_deadline = start_time + 20
        while uncovered_count > 0:
            if time.time() > greedy_deadline:
                break
            # Count how many uncovered words each candidate covers
            counts = uncovered[sphere_idx_2d].sum(axis=1)
            best_idx = int(np.argmax(counts))
            best_count = int(counts[best_idx])
            if best_count <= 0:
                break
            code_indices.append(best_idx)
            uncovered[sphere_idx_2d[best_idx]] = False
            uncovered_count = int(uncovered.sum())

        m = len(code_indices)

        # Phase 2: Vectorized redundancy removal
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in code_indices:
            cover_count[sphere_idx_2d[idx]] += 1

        active = np.ones(m, dtype=bool)
        sphere_idx_matrix = sphere_idx_2d[code_indices].copy()  # shape (m, 17)

        cleanup_deadline = start_time + 35
        for pass_num in range(3000):
            if time.time() > cleanup_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            # For each active codeword, check if all its sphere positions are covered by >= 2 codewords
            active_spheres = sphere_idx_matrix[active_indices]  # (k, 17)
            active_cover_counts = cover_count[active_spheres]    # (k, 17)
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            if not np.any(redundant_mask):
                break
            redundant_local_indices = np.where(redundant_mask)[0]
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            # Vectorized decrement of cover_count
            for ri in redundant_global_indices:
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1

        # Phase 3: Enhanced local search with swap and 2-opt moves
        best_code_size = int(np.sum(active))
        best_active = active.copy()
        best_cover_count = cover_count.copy()
        best_sphere_matrix = sphere_idx_matrix.copy()

        T = 25.0
        cooling_rate = 0.9995
        T_min = 0.1
        sa_deadline = deadline - 15

        # Build a lookup: for each codeword index in code_indices, its position
        idx_to_pos = {}
        for i, ci in enumerate(code_indices):
            idx_to_pos[ci] = i

        no_improve_count = 0
        max_no_improve = 500

        while time.time() < sa_deadline and T > T_min:
            active_count = int(np.sum(active))
            if active_count < 2:
                break

            active_indices = np.where(active)[0]
            
            # Choose move type: 70% remove+add, 30% swap
            move_type = random.random()
            
            if move_type < 0.7:
                # Remove one codeword, then try to add a better one
                remove_pos = active_indices[random.randint(0, len(active_indices) - 1)]
                remove_idx = code_indices[remove_pos]

                # Remove it
                active[remove_pos] = False
                remove_sphere = sphere_idx_matrix[remove_pos]
                cover_count[remove_sphere] -= 1
                new_uncovered = int(np.sum(cover_count == 0))

                if new_uncovered > 0:
                    # Find best candidate: try uncovered words' neighbors
                    uncovered_mask = cover_count == 0
                    uncovered_indices = np.where(uncovered_mask)[0]
                    
                    best_gain = 0
                    best_cand = -1
                    
                    # Try up to 15 random uncovered words, check their sphere for good candidates
                    num_trials = min(15, len(uncovered_indices))
                    if num_trials > 0:
                        trial_indices = random.sample(range(len(uncovered_indices)), num_trials)
                        
                        for ti in trial_indices:
                            uw = int(uncovered_indices[trial_indices[ti]])
                            # The sphere of uw contains uw itself and its neighbors
                            sphere_of_uw = sphere_idx_2d[uw]
                            # For each position in uw's sphere, check how many uncovered it covers
                            for sp in sphere_of_uw:
                                sp = int(sp)
                                gain = int(np.sum(cover_count[sphere_idx_2d[sp]] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_cand = sp

                    if best_cand >= 0 and best_gain > 0:
                        # Add the candidate
                        if best_cand in idx_to_pos:
                            found = idx_to_pos[best_cand]
                            active[found] = True
                            cover_count[sphere_idx_2d[best_cand]] += 1
                        else:
                            code_indices.append(best_cand)
                            idx_to_pos[best_cand] = len(code_indices) - 1
                            m = len(code_indices)
                            sphere_idx_matrix = np.vstack([sphere_idx_matrix, sphere_idx_2d[best_cand]])
                            active = np.append(active, True)
                            cover_count[sphere_idx_2d[best_cand]] += 1

                        new_uncovered_after = int(np.sum(cover_count == 0))
                        if new_uncovered_after == 0:
                            new_size = int(np.sum(active))
                            if new_size < best_code_size:
                                best_code_size = new_size
                                best_active = active.copy()
                                best_cover_count = cover_count.copy()
                                best_sphere_matrix = sphere_idx_matrix.copy()
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                        else:
                            delta_cost = new_uncovered_after
                            if delta_cost > 0 and random.random() < np.exp(-delta_cost / T):
                                no_improve_count += 1
                            else:
                                # Undo the addition
                                if best_cand in idx_to_pos:
                                    found = idx_to_pos[best_cand]
                                    if found != remove_pos:
                                        active[found] = False
                                        cover_count[sphere_idx_2d[best_cand]] -= 1
                    else:
                        active[remove_pos] = True
                        cover_count[remove_sphere] += 1
                        no_improve_count += 1
                else:
                    new_size = int(np.sum(active))
                    if new_size < best_code_size:
                        best_code_size = new_size
                        best_active = active.copy()
                        best_cover_count = cover_count.copy()
                        best_sphere_matrix = sphere_idx_matrix.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            else:
                # Swap: remove one, add another (both active)
                if len(active_indices) >= 2:
                    remove_pos = active_indices[random.randint(0, len(active_indices) - 1)]
                    # Find a different active position to add
                    other_indices = [i for i in active_indices if i != remove_pos]
                    if other_indices:
                        add_pos = other_indices[random.randint(0, len(other_indices) - 1)]
                        remove_idx = code_indices[remove_pos]
                        add_idx = code_indices[add_pos]

                        # Remove
                        active[remove_pos] = False
                        cover_count[sphere_idx_matrix[remove_pos]] -= 1
                        # Add
                        active[add_pos] = True
                        cover_count[sphere_idx_matrix[add_pos]] += 1

                        new_uncovered = int(np.sum(cover_count == 0))
                        if new_uncovered == 0:
                            # Size unchanged, but might be better arrangement
                            no_improve_count += 1
                        else:
                            # Undo
                            active[remove_pos] = True
                            cover_count[sphere_idx_matrix[remove_pos]] += 1
                            active[add_pos] = False
                            cover_count[sphere_idx_matrix[add_pos]] -= 1
                            no_improve_count += 1

            T *= cooling_rate
            
            if no_improve_count > max_no_improve:
                # Restart from best
                active = best_active.copy()
                cover_count = best_cover_count.copy()
                sphere_idx_matrix = best_sphere_matrix.copy()
                no_improve_count = 0
                # Reset idx_to_pos
                idx_to_pos = {}
                for i, ci in enumerate(code_indices):
                    idx_to_pos[ci] = i

        # Restore best state
        active = best_active.copy()
        cover_count = best_cover_count.copy()
        sphere_idx_matrix = best_sphere_matrix.copy()

        # Phase 4: Final aggressive redundancy cleanup
        final_deadline = deadline - 5
        for pass_num in range(5000):
            if time.time() > final_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_spheres = sphere_idx_matrix[active_indices]
            active_cover_counts = cover_count[active_spheres]
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            if not np.any(redundant_mask):
                break
            redundant_local_indices = np.where(redundant_mask)[0]
            redundant_global_indices = active_indices[redundant_local_indices]
            active[redundant_global_indices] = False
            for ri in redundant_global_indices:
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1

        final_indices = [code_indices[i] for i in range(len(active)) if active[i]]
        final_code = [int_to_word(idx) for idx in final_indices]

        # Verify coverage
        verify_cover = np.zeros(total_space, dtype=bool)
        for idx in final_indices:
            verify_cover[sphere_idx_2d[idx]] = True
        if np.all(verify_cover):
            return [list(w) for w in final_code]

        # Fallback: return all words (valid but large)
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    else:
        # General case: use the tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but not minimal)
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
