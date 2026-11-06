    def merge_paths(paths, edges_set):
        if len(paths) <= 1:
            return paths
        
        current_paths = [p[:] for p in paths]
        
        changed = True
        while changed:
            changed = False
            new_paths = []
            used = [False] * len(current_paths)
            
            for i in range(len(current_paths)):
                if used[i]:
                    continue
                pi = current_paths[i]
                if not pi:
                    continue
                vi = set(pi)
                
                merged_with = -1
                new_pi = pi
                
                for j in range(i + 1, len(current_paths)):
                    if used[j]:
                        continue
                    pj = current_paths[j]
                    if not pj:
                        continue
                    vj = set(pj)
                    
                    if vi & vj:
                        continue
                    
                    # Check connections
                    # 1. pi[-1] - pj[0]
                    if tuple(sorted((pi[-1], pj[0]))) in edges_set:
                        new_pi = pi + pj
                        merged_with = j
                        break
                    # 2. pi[-1] - pj[-1]
                    if tuple(sorted((pi[-1], pj[-1]))) in edges_set:
                        new_pi = pi + pj[::-1]
                        merged_with = j
                        break
                    # 3. pi[0] - pj[0]
                    if tuple(sorted((pi[0], pj[0]))) in edges_set:
                        new_pi = pi[::-1] + pj
                        merged_with = j
                        break
                    # 4. pi[0] - pj[-1]
                    if tuple(sorted((pi[0], pj[-1]))) in edges_set:
                        new_pi = pi[::-1] + pj[::-1]
                        merged_with = j
                        break
                
                if merged_with != -1:
                    new_paths.append(new_pi)
                    used[i] = True
                    used[merged_with] = True
                    changed = True
                else:
                    new_paths.append(pi)
                    used[i] = True # Mark as processed, even if not merged, to avoid re-processing?
                    # Actually, if not merged, it stays as is.
                    # We should mark it as "handled" in this pass.
                    # But wait, if I don't merge it now, I shouldn't try to merge it again in the same pass?
                    # No, if I didn't merge it with j, I might merge it with k?
                    # My loop breaks on first merge. If no merge found, it stays.
                    # So I just append it.
            
            current_paths = new_paths
            
        return current_paths
