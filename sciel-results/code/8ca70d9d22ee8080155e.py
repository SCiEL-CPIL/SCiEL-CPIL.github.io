<<<<<<< SEARCH
    M, combo = ans
    combo = _reduce_size(combo, vecs, target, deadline)
    selected = [211 + pos for pos in range(combo.bit_length()) if (combo >> pos) & 1]
=======
    M, combo = ans
    combo = _reduce_size(combo, vecs, target, deadline)
    
    # Local search: try to swap out larger elements for smaller ones
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            if combo == 0:
                break
            max_pos = combo.bit_length() - 1
            # Try to find an alternative representation with smaller max
            # Build basis from positions 0..max_pos-1 (selected only)
            basis_s = {}
            removable = False
            for pos in range(max_pos):
                if not (combo >> pos) & 1:
                    continue
                vec = vectors[pos]
                c = 1 << pos
                while vec:
                    pivot = vec.bit_length() - 1
                    if pivot in basis_s:
                        bv, bc = basis_s[pivot]
                        vec ^= bv
                        c ^= bc
                    else:
                        basis_s[pivot] = (vec, c)
                        break
                if vec == 0 and c != 0:
                    combo ^= c
                    removable = True
                    break
            if not removable:
                # Check if vectors[max_pos] is in span of currently selected others
                vec_check = vectors[max_pos]
                in_span = True
                for pos in range(max_pos):
                    if not (combo >> pos) & 1:
                        continue
                    # reduce vec_check by vectors[pos] if possible
                    # This is tricky; use the basis approach
                    pass
                # Use basis to check
                basis_chk = {}
                for pos in range(max_pos):
                    if not (combo >> pos) & 1:
                        continue
                    vec = vectors[pos]
                    while vec:
                        pivot = vec.bit_length() - 1
                        if pivot in basis_chk:
                            vec ^= basis_chk[pivot]
                        else:
                            basis_chk[pivot] = vec
                            break
                vec_c = vectors[max_pos]
                while vec_c:
                    pivot = vec_c.bit_length() - 1
                    if pivot in basis_chk:
                        vec_c ^= basis_chk[pivot]
                    else:
                        break
                if vec_c == 0:
                    # Can remove max_pos, find the combination
                    basis_comb = {}
                    for pos in range(max_pos):
                        if not (combo >> pos) & 1:
                            continue
                        vec = vectors[pos]
                        c = 1 << pos
                        while vec:
                            pivot = vec.bit_length() - 1
                            if pivot in basis_comb:
                                bv, bc = basis_comb[pivot]
                                vec ^= bv
                                c ^= bc
                            else:
                                basis_comb[pivot] = (vec, c)
                                break
                        if vec == 0 and c != 0:
                            combo ^= c
                            break
                    vec_c2 = vectors[max_pos]
                    c2 = 1 << max_pos
                    while vec_c2:
                        pivot = vec_c2.bit_length() - 1
                        if pivot in basis_comb:
                            bv, bc = basis_comb[pivot]
                            vec_c2 ^= bv
                            c2 ^= bc
                        else:
                            break
                    if vec_c2 == 0:
                        combo ^= c2
                    else:
                        break
                else:
                    break
    
    selected = [211 + pos for pos in range(combo.bit_length()) if (combo >> pos) & 1]
>>>>>>> REPLACE

