"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def _hamming_distance(a, b):
    """Compute Hamming distance between two tuples/lists of integers."""
    return sum(x != y for x, y in zip(a, b))


def _generate_ball(center, q, n, radius):
    """Generate all words within Hamming distance <= radius from center."""
    ball = [center]
    for r in range(1, radius + 1):
        # For each position, change to each other symbol
        for pos in range(n):
            orig = center[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                new_word = list(center)
                new_word[pos] = new_val
                new_word = tuple(new_word)
                ball.append(new_word)
    return set(ball)


def construct(q, n, radius):
    """
    Greedy set-covering with simulated annealing refinement for ternary covering code.
    """
    start_time = time.time()
    time_limit = 115  # leave some buffer before 120s timeout
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = [tuple(word) for word in np.ndindex(*([q] * n))]
    all_words_set = set(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute all balls
    all_balls = {}
    for w in all_words:
        all_balls[w] = _generate_ball(w, q, n, radius)
    
    def greedy_set_cover(uncovered_set, time_budget):
        """Run greedy set cover until uncovered is empty or time budget exceeded."""
        uncovered = set(uncovered_set)
        centers = []
        
        while uncovered:
            if time.time() - start_time > time_budget:
                break
            
            best_center = None
            best_coverage = -1
            best_newly_covered = None
            
            for center in all_words:
                ball = all_balls[center]
                newly_covered = ball & uncovered
                coverage = len(newly_covered)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_center = center
                    best_newly_covered = newly_covered
            
            if best_center is None or best_coverage <= 0:
                break
            
            centers.append(best_center)
            uncovered -= best_newly_covered
        
        # If greedy didn't cover everything, add more centers
        if uncovered:
            for w in list(uncovered):
                centers.append(w)
                uncovered -= all_balls[w]
        
        return centers
    
    def compute_coverage_counts(centers_list):
        """Compute coverage count array for a list of centers."""
        cov = np.zeros(total_words, dtype=np.int32)
        for c in centers_list:
            for w in all_balls[c]:
                idx = word_to_idx[w]
                cov[idx] += 1
        return cov
    
    def is_valid_covering(centers_list):
        """Check if centers_list covers all words."""
        if len(centers_list) == 0:
            return total_words == 0
        cov = compute_coverage_counts(centers_list)
        return np.all(cov > 0)
    
    def simulated_annealing(initial_centers, time_budget):
        """Run simulated annealing to minimize the number of centers."""
        current_centers = list(initial_centers)
        best_centers = list(initial_centers)
        best_count = len(current_centers)
        
        coverage_count = compute_coverage_counts(current_centers)
        
        T = 5.0
        T_min = 0.05
        cooling_rate = 0.995
        
        # Precompute for each center, which words it uniquely covers (coverage_count == 1)
        # This helps in deciding moves
        
        while T > T_min and time.time() - start_time < time_budget:
            move_type = random.random()
            
            if move_type < 0.4 and len(current_centers) > 1:
                # Try removing a random center
                idx = random.randint(0, len(current_centers) - 1)
                c = current_centers[idx]
                
                # Check if removing c still covers everything
                removable = True
                for w in all_balls[c]:
                    idx_w = word_to_idx[w]
                    if coverage_count[idx_w] <= 1:
                        removable = False
                        break
                
                if removable:
                    # Remove c
                    for w in all_balls[c]:
                        idx_w = word_to_idx[w]
                        coverage_count[idx_w] -= 1
                    current_centers.pop(idx)
                    new_count = len(current_centers)
                    
                    if new_count < best_count:
                        best_count = new_count
                        best_centers = list(current_centers)
            
            elif move_type < 0.7 and len(current_centers) >= 2:
                # Try replacing one center with another
                idx_replace = random.randint(0, len(current_centers) - 1)
                c_old = current_centers[idx_replace]
                
                # Find words that are only covered by c_old
                only_covered = []
                for w in all_balls[c_old]:
                    idx_w = word_to_idx[w]
                    if coverage_count[idx_w] == 1:
                        only_covered.append(w)
                
                if not only_covered:
                    # Can just remove c_old
                    for w in all_balls[c_old]:
                        idx_w = word_to_idx[w]
                        coverage_count[idx_w] -= 1
                    current_centers.pop(idx_replace)
                    new_count = len(current_centers)
                    if new_count < best_count:
                        best_count = new_count
                        best_centers = list(current_centers)
                else:
                    # Try to find a new center that covers all only_covered words
                    # Sample some candidate centers
                    for _ in range(20):
                        c_new = random.choice(all_words)
                        if c_new == c_old:
                            continue
                        
                        ball_new = all_balls[c_new]
                        if all(w in ball_new for w in only_covered):
                            # Accept the replacement
                            # Remove c_old
                            for w in all_balls[c_old]:
                                idx_w = word_to_idx[w]
                                coverage_count[idx_w] -= 1
                            
                            # Add c_new
                            for w in ball_new:
                                idx_w = word_to_idx[w]
                                coverage_count[idx_w] += 1
                            
                            current_centers[idx_replace] = c_new
                            break
            
            elif move_type < 0.9 and len(current_centers) >= 3:
                # Try removing two centers and adding one new center
                i, j = random.sample(range(len(current_centers)), 2)
                c1 = current_centers[i]
                c2 = current_centers[j]
                
                # Find words that would become uncovered if we remove both c1 and c2
                # A word is at risk if it's covered by c1 or c2 and has coverage_count == 1 for that center
                # Actually, we need: after removing both, coverage_count[w] > 0 for all w
                
                # Compute what would happen
                temp_cov = coverage_count.copy()
                for w in all_balls[c1]:
                    idx_w = word_to_idx[w]
                    temp_cov[idx_w] -= 1
                for w in all_balls[c2]:
                    idx_w = word_to_idx[w]
                    temp_cov[idx_w] -= 1
                
                # Find words with temp_cov == 0 (would be uncovered)
                uncovered_if_removed = [w for w in all_words if temp_cov[word_to_idx[w]] == 0]
                
                if not uncovered_if_removed:
                    # Can remove both
                    for w in all_balls[c1]:
                        idx_w = word_to_idx[w]
                        coverage_count[idx_w] -= 1
                    for w in all_balls[c2]:
                        idx_w = word_to_idx[w]
                        coverage_count[idx_w] -= 1
                    
                    # Remove both from list (be careful with indices)
                    if i < j:
                        current_centers.pop(j)
                        current_centers.pop(i)
                    else:
                        current_centers.pop(i)
                        current_centers.pop(j)
                    
                    new_count = len(current_centers)
                    if new_count < best_count:
                        best_count = new_count
                        best_centers = list(current_centers)
                else:
                    # Try to find a new center that covers all uncovered_if_removed
                    for _ in range(30):
                        c_new = random.choice(all_words)
                        ball_new = all_balls[c_new]
                        if all(w in ball_new for w in uncovered_if_removed):
                            # Remove c1 and c2, add c_new
                            for w in all_balls[c1]:
                                idx_w = word_to_idx[w]
                                coverage_count[idx_w] -= 1
                            for w in all_balls[c2]:
                                idx_w = word_to_idx[w]
                                coverage_count[idx_w] -= 1
                            for w in ball_new:
                                idx_w = word_to_idx[w]
                                coverage_count[idx_w] += 1
                            
                            if i < j:
                                current_centers.pop(j)
                                current_centers[i] = c_new
                            else:
                                current_centers.pop(i)
                                current_centers[j] = c_new
                            
                            break
            
            # Cool down
            T *= cooling_rate
        
        return best_centers
    
    def local_search_refinement(centers_list):
        """Try to further reduce by greedily removing redundant centers."""
        current = list(centers_list)
        coverage_count = compute_coverage_counts(current)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                c = current[i]
                removable = True
                for w in all_balls[c]:
                    idx_w = word_to_idx[w]
                    if coverage_count[idx_w] <= 1:
                        removable = False
                        break
                
                if removable:
                    for w in all_balls[c]:
                        idx_w = word_to_idx[w]
                        coverage_count[idx_w] -= 1
                    current.pop(i)
                    improved = True
                    break
        
        return current
    
    # Phase 1: Multiple greedy starts with different random orderings
    best_overall = None
    best_overall_count = float('inf')
    
    # Use multiple random seeds for greedy
    for seed in [42, 123, 456, 789, 1000]:
        random.seed(seed)
        
        # Shuffle the order of all_words to get different greedy results
        shuffled_words = list(all_words)
        random.shuffle(shuffled_words)
        
        uncovered = set(all_words_set)
        centers = []
        
        while uncovered:
            if time.time() - start_time > time_limit * 0.15:
                break
            
            best_center = None
            best_coverage = -1
            best_newly_covered = None
            
            for center in shuffled_words:
                ball = all_balls[center]
                newly_covered = ball & uncovered
                coverage = len(newly_covered)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_center = center
                    best_newly_covered = newly_covered
            
            if best_center is None or best_coverage <= 0:
                break
            
            centers.append(best_center)
            uncovered -= best_newly_covered
        
        if uncovered:
            for w in list(uncovered):
                centers.append(w)
                uncovered -= all_balls[w]
        
        if len(centers) < best_overall_count:
            best_overall_count = len(centers)
            best_overall = centers
    
    # Phase 2: Run simulated annealing on the best greedy solution
    if best_overall is not None:
        sa_result = simulated_annealing(best_overall, time_limit * 0.5)
        if len(sa_result) < best_overall_count:
            best_overall = sa_result
            best_overall_count = len(sa_result)
    
    # Phase 3: Local search refinement
    if best_overall is not None:
        refined = local_search_refinement(best_overall)
        if len(refined) < best_overall_count:
            best_overall = refined
            best_overall_count = len(refined)
    
    # Phase 4: Run more SA iterations if time permits
    if best_overall is not None and time.time() - start_time < time_limit * 0.8:
        sa_result2 = simulated_annealing(best_overall, time_limit * 0.9)
        if len(sa_result2) < best_overall_count:
            best_overall = sa_result2
            best_overall_count = len(sa_result2)
    
    # Final local search
    if best_overall is not None:
        final = local_search_refinement(best_overall)
        if len(final) < best_overall_count:
            best_overall = final
            best_overall_count = len(final)
    
    # Verify coverage
    if best_overall is None:
        # Fallback: use all words
        best_overall = list(all_words)
    else:
        covered = set()
        for c in best_overall:
            covered |= all_balls[c]
        if covered != all_words_set:
            # Fallback to greedy
            best_overall = greedy_set_cover(all_words_set, time_limit * 0.9)
    
    # Return as list of lists
    return [list(c) for c in best_overall]
# EVOLVE-BLOCK-END
