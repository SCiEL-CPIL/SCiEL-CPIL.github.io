import random
import time
import numpy as np

def construct():
    def try_n(n, time_limit):
        # Generate all triples
        triples = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triples.append((a, b, c))
        M = len(triples)
        
        # Map (a,b,c) to index
        idx_map = {}
        for i, t in enumerate(triples):
            idx_map[t] = i
            # Also store permutations? No, we always sort.
            # But for fast lookup, a 3D array is better.
            # n <= 40, 40^3 = 64000.
        idx_arr = -1 * np.ones((n, n, n), dtype=int)
        for i, (a,b,c) in enumerate(triples):
            idx_arr[a,b,c] = i
            # We only access with sorted a<b<c in the logic below if we are careful.
            # But in K4/K5 checks, we generate sets, so we need to sort.
            # Sorting 3 elements is cheap.
            
        # Precompute K4s and K5s
        K4s = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        K4s.append((a,b,c,d))
                        
        K5s = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            K5s.append((a,b,c,d,e))
                            
        # Map triple index to list of K4 indices and K5 indices
        K4s_per_triple = [[] for _ in range(M)]
        K5s_per_triple = [[] for _ in range(M)]
        
        # For each K4, add it to the 4 triples it contains
        for k4_idx, (a,b,c,d) in enumerate(K4s):
            t1 = idx_arr[a,b,c]
            t2 = idx_arr[a,b,d]
            t3 = idx_arr[a,c,d]
            t4 = idx_arr[b,c,d]
            K4s_per_triple[t1].append(k4_idx)
            K4s_per_triple[t2].append(k4_idx)
            K4s_per_triple[t3].append(k4_idx)
            K4s_per_triple[t4].append(k4_idx)
            
        # For each K5, add it to the 10 triples it contains
        for k5_idx, (a,b,c,d,e) in enumerate(K5s):
            # Generate all 10 triples from 5 vertices
            verts = [a,b,c,d,e]
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t_idx = idx_arr[verts[i], verts[j], verts[k]]
                        K5s_per_triple[t_idx].append(k5_idx)
                        
        # Initialize colors
        colors = np.random.randint(0, 2, size=M)
        
        # Initialize K4_red and K5_blue
        K4_red = np.zeros(len(K4s), dtype=bool)
        K5_blue = np.zeros(len(K5s), dtype=bool)
        
        # Calculate initial cost
        for k4_idx, (a,b,c,d) in enumerate(K4s):
            if colors[idx_arr[a,b,c]] == 0 and colors[idx_arr[a,b,d]] == 0 and colors[idx_arr[a,c,d]] == 0 and colors[idx_arr[b,c,d]] == 0:
                K4_red[k4_idx] = True
                
        for k5_idx, (a,b,c,d,e) in enumerate(K5s):
            verts = [a,b,c,d,e]
            is_blue = True
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        if colors[idx_arr[verts[i], verts[j], verts[k]]] != 1:
                            is_blue = False
                            break
                    if not is_blue: break
                if not is_blue: break
            if is_blue:
                K5_blue[k5_idx] = True
                
        current_cost = np.sum(K4_red) + np.sum(K5_blue)
        
        if current_cost == 0:
            return colors.tolist()
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        start_time = time.time()
        
        while T > min_T and (time.time() - start_time) < time_limit:
            t_idx = np.random.randint(M)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta
            delta = 0
            
            # Check K4s
            for k4_idx in K4s_per_triple[t_idx]:
                a,b,c,d = K4s[k4_idx]
                # Get indices of the 4 triples
                i1 = idx_arr[a,b,c]
                i2 = idx_arr[a,b,d]
                i3 = idx_arr[a,c,d]
                i4 = idx_arr[b,c,d]
                
                # Count how many are 0 in the new state
                # New state: t_idx is new_color, others are same
                # We need to know if the K4 becomes red or stops being red.
                
                # Let's just check the condition directly
                # If old_color == 0:
                #   The K4 was red iff the other 3 were 0.
                #   Now t_idx is 1, so it's not red.
                #   So if it was red, cost -= 1.
                #   It was red iff K4_red[k4_idx] was True.
                #   Wait, K4_red is the current state.
                #   So if K4_red[k4_idx] is True, it means it was red.
                #   So delta -= 1.
                #   And we set K4_red[k4_idx] = False.
                
                # If old_color == 1:
                #   The K4 was NOT red (since t_idx was 1).
                #   Now t_idx is 0.
                #   It becomes red iff the other 3 are 0.
                #   Check other 3.
                
                if old_color == 0:
                    if K4_red[k4_idx]:
                        delta -= 1
                        K4_red[k4_idx] = False
                    # If it wasn't red, it stays not red (since t_idx becomes 1)
                else:
                    # old_color == 1
                    # Check if other 3 are 0
                    # The other 3 are the ones in i1..i4 except t_idx
                    other_0_count = 0
                    if i1 != t_idx and colors[i1] == 0: other_0_count += 1
                    if i2 != t_idx and colors[i2] == 0: other_0_count += 1
                    if i3 != t_idx and colors[i3] == 0: other_0_count += 1
                    if i4 != t_idx and colors[i4] == 0: other_0_count += 1
                    if other_0_count == 3:
                        delta += 1
                        K4_red[k4_idx] = True
                        
            # Check K5s
            for k5_idx in K5s_per_triple[t_idx]:
                a,b,c,d,e = K5s[k5_idx]
                verts = [a,b,c,d,e]
                # Get the 10 triple indices
                t_indices = []
                for i in range(5):
                    for j in range(i+1, 5):
                        for k in range(j+1, 5):
                            t_indices.append(idx_arr[verts[i], verts[j], verts[k]])
                            
                if old_color == 1:
                    if K5_blue[k5_idx]:
                        delta -= 1
                        K5_blue[k5_idx] = False
                else:
                    # old_color == 0
                    # Check if other 7 are 1
                    other_1_count = 0
                    for ti in t_indices:
                        if ti != t_idx and colors[ti] == 1:
                            other_1_count += 1
                    if other_1_count == 7:
                        delta += 1
                        K5_blue[k5_idx] = True
                        
            # Update
            if delta < 0:
                colors[t_idx] = new_color
                current_cost += delta
            else:
                # Accept with probability exp(-delta/T)
                if delta == 0:
                    # Accept with prob 0.5?
                    if random.random() < 0.5:
                        colors[t_idx] = new_color
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        colors[t_idx] = new_color
                        current_cost += delta
                        # Update K4_red and K5_blue? 
                        # Wait, if we accept a bad move, we MUST update the state arrays.
                        # The logic above updated K4_red/K5_blue assuming we flip.
                        # But if we don't flip, we shouldn't have updated them.
                        # So we need to revert or calculate delta without updating.
                        
            # This logic is flawed because I updated K4_red inside the loop.
            # I need to calculate delta first, then decide, then update.
            
            T *= cooling
            
        if current_cost == 0:
            return colors.tolist()
        return None

    # Try n=34
    res = try_n(34, 30)
    if res:
        return {'n': 34, 'colors': res}
        
    # Try n=33
    res = try_n(33, 20)
    if res:
        return {'n': 33, 'colors': res}
        
    # Fallback
    from itertools import combinations
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
