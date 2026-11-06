    def dfs(idx, current_indices):
        nonlocal best_set, best_residues
        
        if time.time() > deadline:
            return
            
        if len(current_indices) > len(best_set):
            # Solve CSP to get valid residues for this set
            res = solve_csp(current_indices)
            if res:
                best_set = list(current_indices)
                best_residues = res
            
        if len(current_indices) + (n - idx) <= len(best_set):
            return

        # Try adding idx
        current_indices.append(idx)
        dfs(idx + 1, current_indices)
        current_indices.pop()
        
        # Try skipping idx
        dfs(idx + 1, current_indices)
