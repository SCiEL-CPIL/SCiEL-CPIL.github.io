                valid = True
                if u in C:
                    for w in C:
                        if w not in adj[v]:
                            valid = False
                            break
                elif v in C:
                    for w in C:
                        if w not in adj[u]:
                            valid = False
                            break
                else:
                    for w in C:
                        if w not in adj[u] or w not in adj[v]:
                            valid = False
                            break
