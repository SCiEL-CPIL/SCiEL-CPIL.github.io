import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s

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

    def greedy_init(moduli_list):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        covered = np.zeros(N, dtype=np.bool_)
        classes = []
        for m in moduli_list:
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = np.sum(~covered[r:N:m])
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r:N:m] = True
        
        uncovered = np.sum(~covered)
        return classes, int(uncovered)

    def local_search(classes, score, time_limit):
        """Fast local search using cover counts."""
        classes = [c[:] for c in classes]
        score = score
        
        cover_count = np.zeros(N, dtype=np.int16)
        for m, r in classes:
            cover_count[r:N:m] += 1
            
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(len(classes)):
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                
                best_r = r
                best_score = score
                
                # Calculate loss: positions covered only by this class
                current_positions = range(r, N, m)
                loss = np.sum(cover_count[current_positions] == 1)
                
                # Temporarily remove current class
                cover_count[current_positions] -= 1
                
                # Find best new residue
                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_positions = range(new_r, N, m)
                    gain = np.sum(cover_count[new_positions] == 0)
                    
                    new_score = score - gain + loss
                    if new_score < best_score:
                        best_score = new_score
                        best_r = new_r
                
                # Restore current class
                cover_count[current_positions] += 1
                
                if best_r != r:
                    # Apply best move
                    cover_count[range(r, N, m)] -= 1
                    cover_count[range(best_r, N, m)] += 1
                    classes[i] = [m, best_r]
                    score = best_score
                    improved = True

        return classes, score

    # Strategy 1: Greedy with all divisors in increasing order
    best_classes, best_score = greedy_init(divisors)

    # Run local search
    if (time.time() - start_time) < deadline - 10:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 5)

    # Strategy 2: Random restarts with local search
    random.seed(42)
    
    # Try a few random shuffles
    for _ in range(10):
        if (time.time() - start_time) > deadline - 20:
            break
        shuffled = divisors[:]
        random.shuffle(shuffled)
        classes2, score2 = greedy_init(shuffled)
        
        # Only run local search if it has potential or just run it
        # Given LS is fast, we can run it on all
        classes2, score2 = local_search(classes2, score2, deadline - 10)
        
        if score2 < best_score:
            best_classes, best_score = classes2, score2

    # Try descending order
    if (time.time() - start_time) < deadline - 20:
        divs_desc = sorted(divisors, reverse=True)
        classes3, score3 = greedy_init(divs_desc)
        classes3, score3 = local_search(classes3, score3, deadline - 10)
        if score3 < best_score:
            best_classes, best_score = classes3, score3

    # Final local search if time permits
    if (time.time() - start_time) < deadline - 2:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 1)

    return best_classes
