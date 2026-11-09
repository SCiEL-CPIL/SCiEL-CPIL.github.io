        # Build neighbor lists for faster coverage update
        # 6561 * 17 updates is ~100k ops. Fast.
        
        # ... (setup coverage) ...
        
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword
            # We need to iterate in reverse or use a copy?
            # If we remove one, indices shift? No, we use a list of indices.
            
            new_code = []
            for idx in code_indices:
                # Tentatively remove
                # Decrement coverage for neighbors
                for p in neighbors[idx]:
                    coverage[p] -= 1
                    
                # Check if any coverage became 0
                is_valid = True
                for p in neighbors[idx]:
                    if coverage[p] == 0:
                        is_valid = False
                        break
                        
                if is_valid:
                    # Keep it removed
                    changed = True
                else:
                    # Restore
                    for p in neighbors[idx]:
                        coverage[p] += 1
                    new_code.append(idx)
                    
            code_indices = new_code
