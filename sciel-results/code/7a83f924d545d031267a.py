import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s

    N = 51975

    # Generate all divisors of N greater than 1
    divisors = []
    for i in range(1, int(np.sqrt(N)) + 1):
        if N % i == 0:
            if i > 1:
                divisors.append(i)
            other = N // i
            if other != i and other > 1:
                divisors.append(other)
    divisors.sort()
    
    # Precompute indices for each divisor? 
    # No, indices depend on residue. But we can precompute the "step" arrays?
    # Just using range/r::m is fine for N=51975.

    def greedy_init(moduli_list):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        covered = np.zeros(N, dtype=np.uint8)
        classes = []
        for m in moduli_list:
            best_r = 0
            best_count = -1
            for r in range(m):
                # Count uncovered in this class
                # covered[r::m] gives the slice
                cnt = int(np.sum(covered[r::m] == 0))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
            
        uncovered = int(np.sum(covered == 0))
        return classes, uncovered

    def local_search(classes, time_limit):
        """
        Fast local search maintaining cover_count and uncovered_count.
        """
        classes = [c[:] for c in classes]
        
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in classes:
            cover_count[r::m] += 1
            
        uncovered_count = int(np.sum(cover_count == 0))
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(len(classes)):
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                best_r = r
                best_score = uncovered_count
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    
                    cover_count[r::m] -= 1
                    cover_count[new_r::m] += 1
                    new_score = int(np.sum(cover_count == 0))
                    cover_count[new_r::m] -= 1
                    cover_count[r::m] += 1
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_r = new_r

                if best_r != r:
                    # Apply the best change permanently
                    cover_count[r::m] -= 1
                    cover_count[best_r::m] += 1
                    classes[i] = [m, best_r]
                    uncovered_count = best_score
                    improved = True

        return classes, uncovered_count

    # Strategy 1: Greedy with all divisors in increasing order
    best_classes, best_score = greedy_init(divisors)

    # Run local search
    if (time.time() - start_time) < deadline - 10:
        best_classes, best_score = local_search(best_classes, deadline - 5)

    # Strategy 2: Try different orderings
    random.seed(42)
    remaining = deadline - (time.time() - start_time)

    if remaining > 15:
        # Try greedy with shuffled order
        shuffled = divisors[:]
        random.shuffle(shuffled)
        classes2, score2 = greedy_init(shuffled)
        if score2 < best_score:
            best_classes, best_score = classes2, score2
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, deadline - 3)

    if (time.time() - start_time) < deadline - 15:
        # Try with larger moduli first
        divs_desc = sorted(divisors, reverse=True)
        classes3, score3 = greedy_init(divs_desc)
        if score3 < best_score:
            best_classes, best_score = classes3, score3
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, deadline - 3)

    # Strategy 3: Simulated Annealing
    # We need a fast way to evaluate perturbations.
    # We will reuse the cover_count array from the last local search?
    # No, let's initialize it from best_classes.
    
    if (time.time() - start_time) < deadline - 10:
        current_classes = [c[:] for c in best_classes]
        current_score = best_score
        
        cover_count = np.zeros(N, dtype=np.int32)
        for m, r in current_classes:
            cover_count[r::m] += 1
            
        temperature = max(1, best_score // 5)
        
        for iteration in range(500):
            if (time.time() - start_time) > deadline - 5:
                break

            # Perturb: change one residue
            idx = random.randint(0, len(current_classes) - 1)
            m, r = current_classes[idx]
            old_r = r
            new_r = random.randint(0, m - 1)
            
            if new_r == old_r:
                continue

            # Evaluate
            cover_count[old_r::m] -= 1
            cover_count[new_r::m] += 1
            new_score = int(np.sum(cover_count == 0))
            
            delta = new_score - current_score

            if delta < 0 or random.random() < (1.0 / (1.0 + delta / max(1, temperature))):
                current_classes[idx] = [m, new_r]
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in current_classes]
                    # Run local search from new best
                    if (time.time() - start_time) < deadline - 3:
                        best_classes, best_score = local_search(best_classes, best_score, deadline - 2)
                        # Re-sync cover_count with best_classes if we returned from local search
                        # local_search returns classes and score, but doesn't return cover_count.
                        # We should re-initialize cover_count if we do local search.
                        cover_count = np.zeros(N, dtype=np.int32)
                        for mm, rr in best_classes:
                            cover_count[rr::mm] += 1
                        current_classes = [c[:] for c in best_classes]
                        current_score = best_score
            else:
                # Revert
                cover_count[new_r::m] -= 1
                cover_count[old_r::m] += 1

            # Cool down
            temperature = max(1, temperature * 0.95)

    # Final local search if time permits
    if (time.time() - start_time) < deadline - 2:
        best_classes, best_score = local_search(best_classes, deadline - 1)

    return best_classes
