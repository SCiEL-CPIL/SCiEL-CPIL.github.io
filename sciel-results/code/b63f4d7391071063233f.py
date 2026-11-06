import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    max_time = 140.0  # Leave 10s margin for 150s limit
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # We will try to build up to n=40
    # Using a randomized greedy approach
    
    for attempt in range(10000): # Cap attempts, but also check time
        if time.time() - start_time > max_time:
            break
        
        # Random seed for this attempt
        random.seed(attempt)
        
        n_max = 40
        # C[i][j][k] where i<j<k
        # We can use a list of lists or a dict. 
        # For speed, a 3D numpy array or a flat list with index calculation is good.
        # Index for (i,j,k) in lex order of all triples? 
        # It's easier to just store in a dict or a 3D list if n is small.
        # n=40 is small. 40*40*40 = 64000. 
        # Let's use a 3D list C[i][j][k] = -1, 0, 1.
        C = [[[-1 for _ in range(n_max)] for _ in range(n_max)] for _ in range(n_max)]
        
        success = True
        current_n = 0
        
        for v in range(n_max):
            # Add vertex v
            # We need to color triples (i, j, v) for 0 <= i < j < v
            for i in range(v):
                for j in range(i + 1, v):
                    # Determine color for (i, j, v)
                    
                    # Check if Red (0) is forbidden
                    red_forbidden = False
                    # Red K4: exists k < i such that (k,i,j), (k,i,v), (k,j,v) are Red
                    for k in range(i):
                        if C[k][i][j] == 0 and C[k][i][v] == 0 and C[k][j][v] == 0:
                            red_forbidden = True
                            break
                    
                    # Check if Blue (1) is forbidden
                    blue_forbidden = False
                    # Blue K5: exists k, l < i such that all 9 other triples in {k,l,i,j,v} are Blue
                    # Triples:
                    # (k,l,i), (k,l,j), (k,l,v)
                    # (k,i,j), (k,i,v), (k,j,v)
                    # (l,i,j), (l,i,v), (l,j,v)
                    for k in range(i):
                        for l in range(k + 1, i):
                            if (C[k][l][i] == 1 and C[k][l][j] == 1 and C[k][l][v] == 1 and
                                C[k][i][j] == 1 and C[k][i][v] == 1 and C[k][j][v] == 1 and
                                C[l][i][j] == 1 and C[l][i][v] == 1 and C[l][j][v] == 1):
                                blue_forbidden = True
                                break
                        if blue_forbidden:
                            break
                    
                    if red_forbidden and blue_forbidden:
                        # Stuck
                        success = False
                        break
                    elif red_forbidden:
                        C[i][j][v] = 1
                    elif blue_forbidden:
                        C[i][j][v] = 0
                    else:
                        # Both are safe. 
                        # Heuristic: Random choice, or balance?
                        # Random choice introduces diversity for restarts.
                        C[i][j][v] = random.randint(0, 1)
            
            if not success:
                break
            
            current_n = v + 1
            if current_n > best_n:
                best_n = current_n
                # Extract colors in lex order
                best_colors = []
                for a in range(current_n):
                    for b in range(a + 1, current_n):
                        for c in range(b + 1, current_n):
                            best_colors.append(C[a][b][c])
        
        # If we reached n_max without failure, we can stop trying to improve n (it's maxed out)
        if best_n == n_max:
            break

    return {'n': best_n, 'colors': best_colors}
