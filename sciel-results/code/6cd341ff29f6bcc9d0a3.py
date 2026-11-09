import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflict masks
    # conflicts[i] is a bitmask where bit j is set if numbers[i] and numbers[j] conflict
    conflicts = [0] * N
    for i in range(N):
        a = numbers[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = numbers[j]
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflicts[i] = mask

    # Tomita's Algorithm for Maximum Independent Set
    # We work with bitmasks for speed

    best_size = 0
    best_solution_mask = 0

    # Helper: Greedy Coloring
    def greedy_color(P):
        U = P
        colors = [0] * N
        max_color = 0
        
        verts_in_P = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            verts_in_P.append(idx)
            temp ^= lsb
        
        for v in verts_in_P:
            neighbor_colors = 0
            neigh_mask = conflicts[v] & P
            temp_neigh = neigh_mask
            while temp_neigh:
                lsb = temp_neigh & (-temp_neigh)
                u = lsb.bit_length() - 1
                if colors[u] > 0:
                    neighbor_colors |= (1 << colors[u])
                temp_neigh ^= lsb
            
            c = 1
            while (neighbor_colors >> c) & 1:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        cv_list = [(colors[v], v) for v in verts_in_P]
        cv_list.sort(key=lambda x: -x[0])
        sorted_verts = [v for c, v in cv_list]
        
        return sorted_verts, max_color, colors

    def expand(X, P, bound):
        nonlocal best_size, best_solution_mask
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if X.bit_count() > best_size:
                best_size = X.bit_count()
                best_solution_mask = X
            return
        
        if X.bit_count() + bound <= best_size:
            return
        
        sorted_verts, max_color, colors = greedy_color(P)
        
        if max_color < bound:
            bound = max_color
            
        if X.bit_count() + bound <= best_size:
            return
        
        # Process vertices in reverse order of the sorted list (highest color first)
        for i in range(len(sorted_verts) - 1, -1, -1):
            v = sorted_verts[i]
            
            if not (P & (1 << v)):
                continue
                
            new_P = P & ~(1 << v) & ~conflicts[v]
            new_X = X | (1 << v)
            
            if i + 1 < len(sorted_verts):
                next_bound = colors[sorted_verts[i+1]] 
            else:
                next_bound = 0
                
            expand(new_X, new_P, next_bound)
            
            P = P & ~(1 << v)
            
            if X.bit_count() + bound <= best_size:
                break

    # Initial P is all vertices
    initial_P = (1 << N) - 1
    initial_X = 0
    initial_bound = N
    
    # Greedy MIS for initial solution
    def greedy_mis(P):
        selected = 0
        while P:
            best_v = -1
            min_deg = N + 1
            temp = P
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (conflicts[v] & P).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v != -1:
                selected |= (1 << best_v)
                P = P & ~(1 << best_v) & ~conflicts[best_v]
        return selected

    greedy_sol = greedy_mis(initial_P)
    best_size = greedy_sol.bit_count()
    best_solution_mask = greedy_sol
    
    # Run Tomita
    expand(initial_X, initial_P, initial_bound)
    
    # Convert best_solution_mask to list of values
    result = []
    temp = best_solution_mask
    while temp:
        lsb = temp & (-temp)
        idx = lsb.bit_length() - 1
        result.append(numbers[idx])
        temp ^= lsb
        
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a*b) % (a+b) == 0:
                valid = False
    print(valid)
