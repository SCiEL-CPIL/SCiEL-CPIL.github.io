    # 3. Local search on best_pairs
    if best_pairs:
        current_pairs = best_pairs[:]
        used_mods = set(m for _, m in current_pairs)
        cur_mod_indices = [mod_index[m] for _, m in current_pairs]
        mod_to_res = {m: r for r, m in current_pairs}
        
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            
            added = False
            candidates = [i for i in range(n_mod) if moduli[i] not in used_mods]
            random.shuffle(candidates)
            
            for c_idx in candidates:
                compatible = True
                for sel_idx in cur_mod_indices:
                    if gcd_table[c_idx][sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                m_new = moduli[c_idx]
                constraints = []
                for sel_idx in cur_mod_indices:
                    g = gcd_table[c_idx][sel_idx]
                    if g > 1:
                        r_sel = mod_to_res[moduli[sel_idx]]
                        constraints.append((g, r_sel % g))
                
                found_r = None
                for r in range(m_new):
                    valid = True
                    for (g, r_mod) in constraints:
                        if r % g == r_mod:
                            valid = False
                            break
                    if valid:
                        found_r = r
                        break
                
                if found_r is not None:
                    current_pairs.append((found_r, m_new))
                    used_mods.add(m_new)
                    cur_mod_indices.append(c_idx)
                    mod_to_res[m_new] = found_r
                    added = True
                    break
            
            if added:
                if len(current_pairs) > best_count:
                    best_count = len(current_pairs)
                    best_pairs = current_pairs[:]
            else:
                # Swap
                if len(current_pairs) > 5:
                    rm_idx = random.randint(0, len(current_pairs) - 1)
                    r_old, m_old = current_pairs[rm_idx]
                    new_pairs = [p for i, p in enumerate(current_pairs) if i != rm_idx]
                    new_used = set(m for _, m in new_pairs)
                    new_mod_indices = [mod_index[m] for _, m in new_pairs]
                    new_mod_to_res = {m: r for r, m in new_pairs}
                    
                    unused = [i for i in range(n_mod) if moduli[i] not in new_used]
                    if unused:
                        c_idx = random.choice(unused)
                        m_new = moduli[c_idx]
                        compatible = True
                        for sel_idx in new_mod_indices:
                            if gcd_table[c_idx][sel_idx] == 1:
                                compatible = False
                                break
                        if compatible:
                            constraints = []
                            for sel_idx in new_mod_indices:
                                g = gcd_table[c_idx][sel_idx]
                                if g > 1:
                                    r_sel = new_mod_to_res[moduli[sel_idx]]
                                    constraints.append((g, r_sel % g))
                            
                            found_r = None
                            for r in range(m_new):
                                valid = True
                                for (g, r_mod) in constraints:
                                    if r % g == r_mod:
                                        valid = False
                                        break
                                if valid:
                                    found_r = r
                                    break
                            
                            if found_r is not None:
                                current_pairs = new_pairs + [(found_r, m_new)]
                                used_mods = new_used | {m_new}
                                cur_mod_indices = new_mod_indices + [c_idx]
                                mod_to_res = new_mod_to_res
                                mod_to_res[m_new] = found_r
