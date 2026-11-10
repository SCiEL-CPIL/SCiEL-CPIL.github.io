            for i, C in enumerate(cliques):
                C_set = clique_sets[i]
                
                u_in = u in C_set
                v_in = v in C_set
                
                if u_in and v_in:
                    placed = True
                    break
                    
                if u_in:
                    if all(v in adj[w] for w in C_set):
                        # Add v
                        C.append(v)
                        C_set.add(v)
                        placed = True
                        break
                elif v_in:
                    if all(u in adj[w] for w in C_set):
                        # Add u
                        C.append(u)
                        C_set.add(u)
                        placed = True
                        break
                else:
                    if all(u in adj[w] and v in adj[w] for w in C_set):
                        # Add u and v
                        C.append(u)
                        C.append(v)
                        C_set.add(u)
                        C_set.add(v)
                        placed = True
                        break
