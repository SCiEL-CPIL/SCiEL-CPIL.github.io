            # Local Search Phase
            # Try to improve min degree
            max_local_steps = 50
            for _ in range(max_local_steps):
                if time.time() > deadline:
                    break
                min_deg = min(deg)
                if min_deg >= min_deg_needed:
                    break
                
                # Find vertex with min degree (randomize if tie)
                min_val = min(deg)
                u_candidates = [i for i in range(target_n) if deg[i] == min_val]
                u = random.choice(u_candidates)
                
                # Find a non-neighbor v
                non_neighbors = [v for v in range(target_n) if v != u and not (adj[u] & (1 << v))]
                if not non_neighbors:
                    break # u is connected to everyone, but deg < target? Impossible if n is small, but possible if target is high.
                random.shuffle(non_neighbors)
                
                success = False
                for v in non_neighbors:
                    if can_add(u, v):
                        add_edge(u, v)
                        success = True
                        break
                    else:
                        # Try swap
                        common = adj[u] & adj[v]
                        if common.bit_count() >= 2:
                            cn_list = []
                            c = common
                            while c:
                                a_bit = c & -c
                                a = a_bit.bit_length() - 1
                                c ^= a_bit
                                cn_list.append(a)
                            
                            for w in cn_list:
                                if deg[w] > min_val:
                                    # Remove (v, w) and add (u, v)
                                    # Check if (v, w) is actually an edge
                                    if adj[v] & (1 << w):
                                        remove_edge(v, w)
                                        add_edge(u, v)
                                        success = True
                                        break
                            if success:
                                break
                if not success:
                    # If no improvement in this step, break to avoid infinite loop or useless cycles
                    # Or continue to try other u?
                    # Given the small size, breaking is safer to save time for next restart
                    break

