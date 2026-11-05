import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if n <= 6:
        # Standard recursive construction is optimal for n <= 6
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Step 1: Build S_6 standard construction to get the list of permutations
        word6 = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word6) - m + 2):
                window = tuple(word6[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word6 = new
        
        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word6) - 5):
            window = tuple(word6[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms) # Should be 720
        
        # Precompute overlap matrix
        # M[i][j] = length of longest suffix of perms[i] matching prefix of perms[j]
        # Max overlap is 5
        M = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j:
                    M[i][j] = 6
                    continue
                q = perms[j]
                best = 0
                # Check k from 5 down to 1
                for k in range(5, 0, -1):
                    if p[-k:] == q[:k]:
                        best = k
                        break
                M[i][j] = best

        # Objective: Maximize sum of overlaps in a Hamiltonian path
        # Length of superperm = 7! + 6! + ... + 1! - Sum(Overlaps)
        # Base length = 5913.
        
        def calc_score(path):
            total = 0
            for i in range(len(path) - 1):
                total += M[path[i]][path[i+1]]
            return total

        def build_word_from_path(path):
            new_word = []
            for idx in path:
                p = perms[idx]
                block = list(p) + [7] + list(p)
                overlap = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word)-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            return new_word

        # Simulated Annealing
        # Start with a random path or the standard order
        # Standard order is the order in which they appeared in word6
        current_path = list(range(N))
        current_score = calc_score(current_path)
        
        best_path = current_path[:]
        best_score = current_score
        
        # Simulated Annealing Parameters
        T = 5.0  # Initial temperature (max overlap is 5, so this allows large jumps)
        cooling = 0.999
        min_T = 0.001
        max_iterations = 2000000 # Cap iterations, but we are time-bound
        
        # Use a fast evaluation for moves
        # 2-opt: reverse segment [i, j]
        # Delta score = 
        #   - M[path[i-1]][path[i]] - M[path[j]][path[j+1]]
        #   + M[path[i-1]][path[j]] + M[path[i]][path[j+1]]
        #   (Handle boundaries i=0, j=N-1)
        
        # Or-opt: move segment [i, i+k-1] to position j
        # This is more complex to delta-evaluate quickly, so we might stick to 2-opt and 3-opt or use full recalculation for Or-opt if fast enough.
        # N=720. Full recalculation is O(N). 2-opt is O(1).
        
        # Let's use 2-opt primarily, and occasionally Or-opt with full recalculation.
        
        improved = True
        while time.time() < deadline - 2:
            if T < min_T:
                # Restart with random path if stuck? Or just break.
                # Let's try a few restarts if time permits
                if random.random() < 0.01:
                    current_path = list(range(N))
                    random.shuffle(current_path)
                    current_score = calc_score(current_path)
                    T = 5.0
                else:
                    break
            
            # 2-opt move
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Calculate delta score for reversing [i, j]
            # Old edges: (i-1, i) and (j, j+1)
            # New edges: (i-1, j) and (i, j+1)
            
            score_change = 0
            
            # Left edge
            if i == 0:
                # No left edge
                pass
            else:
                score_change -= M[current_path[i-1]][current_path[i]]
                score_change += M[current_path[i-1]][current_path[j]]
                
            # Right edge
            if j == N - 1:
                # No right edge
                pass
            else:
                score_change -= M[current_path[j]][current_path[j+1]]
                score_change += M[current_path[i]][current_path[j+1]]
                
            # Internal edges of the reversed segment
            # Old: M[path[i]][path[i+1]] ... M[path[j-1]][path[j]]
            # New: M[path[j]][path[j-1]] ... M[path[i+1]][path[i]]
            # Since M is not symmetric, we need to sum the reverse.
            # Sum_old = sum(M[path[k]][path[k+1]] for k in i..j-1)
            # Sum_new = sum(M[path[k+1]][path[k]] for k in i..j-1)
            
            # For N=720, summing up to 720 terms is fast.
            sum_old = 0
            sum_new = 0
            for k in range(i, j):
                sum_old += M[current_path[k]][current_path[k+1]]
                sum_new += M[current_path[k+1]][current_path[k]]
            
            score_change += sum_new - sum_old
            
            if score_change > 0 or random.random() < pow(2.71828, score_change / T):
                # Accept move
                current_path[i:j+1] = reversed(current_path[i:j+1])
                current_score += score_change
                
                if current_score > best_score:
                    best_score = current_score
                    best_path = current_path[:]
            
            T *= cooling
            
        # Final polish: 2-opt local search on best_path
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N - 1):
                for j in range(i + 1, N):
                    # Check if reversing i..j improves
                    score_change = 0
                    if i > 0:
                        score_change -= M[best_path[i-1]][best_path[i]]
                        score_change += M[best_path[i-1]][best_path[j]]
                    if j < N - 1:
                        score_change -= M[best_path[j]][best_path[j+1]]
                        score_change += M[best_path[i]][best_path[j+1]]
                    
                    sum_old = 0
                    sum_new = 0
                    for k in range(i, j):
                        sum_old += M[best_path[k]][best_path[k+1]]
                        sum_new += M[best_path[k+1]][best_path[k]]
                    score_change += sum_new - sum_old
                    
                    if score_change > 0:
                        best_path[i:j+1] = reversed(best_path[i:j+1])
                        best_score += score_change
                        improved = True
                        break
                if improved:
                    break

        final_word = build_word_from_path(best_path)
        return "".join(map(str, final_word))

    # Fallback for n > 7
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            min_len = min(len(new), len(block))
            for o in range(min_len, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
