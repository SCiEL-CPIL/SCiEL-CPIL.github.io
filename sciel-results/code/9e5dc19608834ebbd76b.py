            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = []
                for u, v in remaining:
                    if u == current and v not in visited:
                        neighbors.append(v)
                    elif v == current and u not in visited:
                        neighbors.append(u)
                
                if not neighbors:
                    break
                
                # Choose next neighbor with lowest degree, then lowest ID
                candidates = []
                for nb in neighbors:
                    deg = 0
                    for u, v in remaining:
                        if u == nb or v == nb:
                            deg += 1
                    candidates.append((deg, nb))
                
                candidates.sort()
                best_next = candidates[0][1]
                
                edge_to_remove = tuple(sorted((current, best_next)))
                remaining.remove(edge_to_remove)
                
                path.append(best_next)
                visited.add(best_next)
                current = best_next
