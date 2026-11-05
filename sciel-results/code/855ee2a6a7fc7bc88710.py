import time
import random
import sys
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 115

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
        # Generate all 720 permutations of 1..6
        # Using the recursive construction order is a good start, 
        # but we will search over all permutations.
        # Actually, the set of permutations is fixed. The order is what we search.
        
        # Generate permutations
        perms = []
        def generate(perms, current, used):
            if len(current) == 6:
                perms.append(tuple(current))
                return
            for i in range(1, 7):
                if not used[i]:
                    used[i] = True
                    current.append(i)
                    generate(perms, current, used)
                    current.pop()
                    used[i] = False
        generate(perms, [], [False]*7)
        
        N = len(perms) # 720
        
        # Map permutation tuple to index
        perm_to_idx = {p: i for i, p in enumerate(perms)}
        
        # Precompute overlap matrix M
        # M[i][j] = overlap between perms[i] and perms[j]
        # overlap: longest suffix of i matching prefix of j
        M = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j:
                    M[i][j] = 6
                    continue
                q = perms[j]
                # Max overlap is 5
                # Check k from 5 down to 0
                # p[-k:] == q[:k]
                # For k=5: p[-5:] vs q[:5]
                # For k=4: p[-4:] vs q[:4]
                # ...
                # Since permutations are distinct, overlap cannot be 6 unless i==j.
                # Max overlap for distinct perms is 5.
                # Actually, if p = 123456, q = 234561, overlap is 5 (23456).
                # If p = 123456, q = 345612, overlap is 4 (3456).
                
                # Efficient check
                # We can precompute prefixes and suffixes?
                # Or just loop k.
                best = 0
                for k in range(5, 0, -1):
                    if p[-k:] == q[:k]:
                        best = k
                        break
                M[i][j] = best

        # Objective: Maximize sum of M[path[i]][path[i+1]]
        # Length = 9360 - sum_overlaps
        
        # Initial path: Greedy
        # Start with a random permutation
        # At each step, pick the unvisited node with max overlap
        
        def greedy_init():
            visited = [False] * N
            path = []
            start = random.randint(0, N - 1)
            visited[start] = True
            path.append(start)
            current = start
            
            for _ in range(N - 1):
                best_next = -1
                best_overlap = -1
                # Find unvisited node with max overlap
                # M[current, j]
                row = M[current]
                for j in range(N):
                    if not visited[j] and row[j] > best_overlap:
                        best_overlap = row[j]
                        best_next = j
                
                if best_next == -1:
                    # Should not happen if graph is complete
                    # But if all overlaps are 0, pick any
                    for j in range(N):
                        if not visited[j]:
                            best_next = j
                            break
                
                visited[best_next] = True
                path.append(best_next)
                current = best_next
                
            return path

        current_path = greedy_init()
        
        # Calculate initial score
        def calc_score(path):
            total = 0
            for i in range(len(path) - 1):
                total += M[path[i]][path[i+1]]
            return total

        current_score = calc_score(current_path)
        best_path = current_path[:]
        best_score = current_score
        
        # Simulated Annealing
        T = 5.0
        cooling = 0.9995
        min_T = 0.001
        no_improve_counter = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 2:
            if T < min_T:
                break
            
            # 2-opt move
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            # Calculate delta
            # Old edges: (i-1, i), (j, j+1)
            # New edges: (i-1, j), (i, j+1)
            # Internal edges reversed
            
            score_change = 0
            
            if i > 0:
                score_change -= M[current_path[i-1]][current_path[i]]
                score_change += M[current_path[i-1]][current_path[j]]
            
            if j < N - 1:
                score_change -= M[current_path[j]][current_path[j+1]]
                score_change += M[current_path[i]][current_path[j+1]]
            
            # Internal
            # Sum M[path[k]][path[k+1]] for k in i..j-1
            # vs Sum M[path[k+1]][path[k]] for k in i..j-1
            
            sum_old = 0
            sum_new = 0
            for k in range(i, j):
                sum_old += M[current_path[k]][current_path[k+1]]
                sum_new += M[current_path[k+1]][current_path[k]]
            
            score_change += sum_new - sum_old
            
            if score_change > 0 or random.random() < np.exp(score_change / T):
                # Accept
                current_path[i:j+1] = reversed(current_path[i:j+1])
                current_score += score_change
                
                if current_score > best_score:
                    best_score = current_score
                    best_path = current_path[:]
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                
                if no_improve_counter > max_no_improve:
                    # Restart with greedy or random
                    current_path = greedy_init()
                    current_score = calc_score(current_path)
                    T = 5.0
                    no_improve_counter = 0
            
            T *= cooling

        # Final 2-opt polish
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N - 1):
                for j in range(i + 1, N):
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

        # Build word
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
