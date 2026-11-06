    def optimize_adj(adj, n, time_limit):
        start_time = time.time()
        temp = 1.0
        cooling = 0.9999
        current_energy = count_mono_triangles(adj, n)
        
        if current_energy == 0:
            return adj, 0
            
        best_energy = current_energy
        best_adj = [row[:] for row in adj]
        
        while time.time() - start_time < time_limit:
            if temp < 0.001:
                break
                
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if j < i:
                i, j = j, i
                
            old_c = adj[i][j]
            new_c = random.choice([0, 1, 2, 3])
            if new_c == old_c:
                continue
                
            delta = 0
            row_i = adj[i]
            row_j = adj[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                if row_i[k] == old_c and row_j[k] == old_c:
                    delta -= 1
                if row_i[k] == new_c and row_j[k] == new_c:
                    delta += 1
                    
            if delta <= 0:
                row_i[j] = new_c
                row_j[i] = new_c
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = [row[:] for row in adj]
                    if best_energy == 0:
                        return best_adj, 0
            else:
                if delta < 50 * temp and random.random() < math.exp(-delta / temp):
                    row_i[j] = new_c
                    row_j[i] = new_c
                    current_energy += delta
                    
            temp *= cooling
            
        return best_adj, best_energy
