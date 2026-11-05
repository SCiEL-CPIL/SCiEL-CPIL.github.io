<<<<<<< SEARCH
    # ---------- Phase 2: Local Search ----------
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1
=======
    # ---------- Phase 2: Local Search ----------
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1
    
    # Additional structured local search: try to replace pairs of codewords
    # in the same plane with a single codeword if possible
    def try_merge_plane_pair():
        """Try to find two codewords in the same plane that can be merged."""
        # Group code by plane
        plane_code = {}
        for c in code:
            pk = point_to_plane[c]
            if pk not in plane_code:
                plane_code[pk] = []
            plane_code[pk].append(c)
        
        for pk, cwds in plane_code.items():
            if len(cwds) < 2:
                continue
            # Try removing each codeword in the pair
            for i in range(len(cwds)):
                for j in range(i + 1, len(cwds)):
                    c1, c2 = cwds[i], cwds[j]
                    # Check if removing both leaves everything covered
                    temp = counts.copy()
                    temp[neighbors_list[c1]] -= 1
                    temp[neighbors_list[c2]] -= 1
                    if np.min(temp) >= 0:
                        # Both can be removed
                        code.remove(c1)
                        code_set.remove(c1)
                        counts[neighbors_list[c1]] -= 1
                        code.remove(c2)
                        code_set.remove(c2)
                        counts[neighbors_list[c2]] -= 1
                        return True
                    # Try replacing both with a single codeword
                    # The union of points only covered by c1 or c2
                    only_c1 = [x for x in neighbors_list[c1] if counts[x] == 1]
                    only_c2 = [x for x in neighbors_list[c2] if counts[x] == 1]
                    needed = set(only_c1) | set(only_c2)
                    if not needed:
                        continue
                    # Find a codeword that covers all needed points
                    first_p = list(needed)[0]
                    for w in neighbor_sets[first_p]:
                        if w in code_set and w != c1 and w != c2:
                            continue
                        ok = True
                        for k in needed:
                            if k not in neighbor_sets[w]:
                                ok = False
                                break
                        if ok:
                            code.remove(c1)
                            code_set.remove(c1)
                            counts[neighbors_list[c1]] -= 1
                            code.remove(c2)
                            code_set.remove(c2)
                            counts[neighbors_list[c2]] -= 1
                            if w not in code_set:
                                code.append(w)
                                code_set.add(w)
                                counts[neighbors_list[w]] += 1
                            return True
        return False
>>>>>>> REPLACE

