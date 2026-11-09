<<<<<<< SEARCH
    # 3. Final refinement pass on the best family found
    if best_family and time.time() < deadline - 1.0:
        # Rebuild inter_maps for best_family
        F = list(best_family)
        inter_maps = []
        for i in range(len(F)):
            inter_maps.append({})
        
        for i in range(len(F)):
            for j in range(i + 1, len(F)):
                s = F[i] & F[j]
                if s not in inter_maps[i]:
                    inter_maps[i][s] = []
                inter_maps[i][s].append(j)
                if s not in inter_maps[j]:
                    inter_maps[j][s] = []
                inter_maps[j][s].append(i)
        
        # Try local refinement
        refined_F, refined_inter_maps = local_refine(F, inter_maps, deadline - 0.5)
        if len(refined_F) > best_size:
            best_family = refined_F
            best_size = len(refined_F)
=======
    # 3. Final refinement pass on the best family found
    if best_family and time.time() < deadline - 1.0:
        # Rebuild inter_maps for best_family
        F = list(best_family)
        inter_maps = rebuild_inter_maps(F)
        
        # Try multiple rounds of local refinement
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            refined_F, refined_inter_maps = local_refine(F, inter_maps, deadline - 0.5)
            if len(refined_F) > len(F):
                F = refined_F
                inter_maps = refined_inter_maps
            else:
                break
        
        if len(F) > best_size:
            best_family = F
            best_size = len(F)
>>>>>>> REPLACE
