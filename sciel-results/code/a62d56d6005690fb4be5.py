            nodes_visited = 0
            max_nodes = 50000
            
            def bron_kerbosch_uncovered(R, P, X, current_covered):
                nonlocal best_clique, best_score, stop_search, nodes_visited
                
                if stop_search:
                    return
                nodes_visited += 1
                if nodes_visited > max_nodes:
                    stop_search = True
                    return
                # ... rest of logic
