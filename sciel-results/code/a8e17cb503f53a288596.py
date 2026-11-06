    def local_search(adj_init, n, max_time):
        best_adj = adj_init[:]
        best_cost = count_forbidden(adj_init, n)[0] + count_forbidden(adj_init, n)[1]
        
        start = time.time()
        while time.time() < start + max_time:
            # Restart if stuck or periodically
            # Simple: just run hill climbing from current state
            # Or: periodically randomize
            adj = best_adj[:]
            current_cost = best_cost
            
            # Hill climb
            while time.time() < start + max_time:
                if current_cost == 0:
                    return adj, 0
                
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                # Flip
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
                
                new_cost = count_forbidden(adj, n)[0] + count_forbidden(adj, n)[1]
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_adj = adj[:]
                else:
                    # Revert
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                    
                # Optional: Random restart if no improvement for a while?
                # For simplicity, just keep going.
                
        return best_adj, best_cost
