            # Try 0
            assign_copy = assignment[:]
            assign_copy[var] = 0
            ok, new_assign, new_edges = propagate(assign_copy, edge_list)
            if ok:
                if backtrack(new_assign, new_edges):
                    return True
            
            # Try 1
            assign_copy = assignment[:]
            assign_copy[var] = 1
            ok, new_assign, new_edges = propagate(assign_copy, edge_list)
            if ok:
                if backtrack(new_assign, new_edges):
                    return True
