    def solve_min_conflicts(n, time_limit):
        # Initialize random coloring
        colors = [[-1] * n for _ in range(n)]
        adj = [[0] * n for _ in range(4)]
        
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                colors[i][j] = c
                colors[j][i] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        # Min-Conflicts Loop
        steps = 0
        max_steps_per_restart = 100
        
        while time.time() < time_limit:
            if steps > max_steps_per_restart:
                # Restart
                for i in range(n):
                    for j in range(i + 1, n):
                        old_c = colors[i][j]
                        adj[old_c][i] &= ~(1 << j)
                        adj[old_c][j] &= ~(1 << i)
                        c = random.randint(0, 3)
                        colors[i][j] = c
                        colors[j][i] = c
                        adj[c][i] |= (1 << j)
                        adj[c][j] |= (1 << i)
                steps = 0
                continue
                
            # Check for C6
            c6 = find_c6_full(adj, n)
            if not c6:
                return colors
            
            # Pick a random edge from the found C6
            edge_idx = random.randint(0, len(c6) - 1)
            u, v = c6[edge_idx]
            
            # Change its color to a random other color
            old_c = colors[u][v]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Update adj and colors
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            colors[u][v] = new_c
            colors[v][u] = new_c
            
            steps += 1
            
        return None
