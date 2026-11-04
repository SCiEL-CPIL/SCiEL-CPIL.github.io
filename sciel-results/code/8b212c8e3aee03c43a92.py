<<<<<<< SEARCH
    # Merge passes: try to combine pairs of terms into a single rank-1 term
    max_passes = 20
    for pass_num in range(max_passes):
        if time.time() > deadline - 10:
            break
        merged_any = False
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            for i in range(len(terms)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(terms)):
                    if time.time() > deadline - 5:
                        break
                    t1 = term_to_sparse(*terms[i])
                    t2 = term_to_sparse(*terms[j])
                    combined = {}
                    for k, v in t1.items():
                        combined[k] = v
                    for k, v in t2.items():
                        combined[k] = combined.get(k, 0) + v
                    combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}

                    if not combined:
                        # They cancel out entirely - remove both
                        terms.pop(j)
                        terms.pop(i)
                        merged_any = True
                        changed = True
                        break
                    continue

                    result = is_rank1_and_extract(combined)
                    if result is not None:
                        a_frac, b_frac, c_frac = result
                        _, a_int, b_int, c_int = scale_to_ints(a_frac, b_frac, c_frac)
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_int, b_int, c_int))
                        merged_any = True
                        changed = True
                        break
                if changed:
                    break
        if not merged_any:
            break

    # Try triple merges
    for pass_num in range(10):
        if time.time() > deadline - 5:
            break
        found = False
        for i in range(len(terms)):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 3:
                    break
                for k in range(j + 1, len(terms)):
                    if time.time() > deadline - 3:
                        break
                    t1 = term_to_sparse(*terms[i])
                    t2 = term_to_sparse(*terms[j])
                    t3 = term_to_sparse(*terms[k])
                    combined = {}
                    for tt in [t1, t2, t3]:
                        for key, val in tt.items():
                            combined[key] = combined.get(key, 0) + val
                    combined = {key: val for key, val in combined.items() if abs(val) > 1e-12}

                    if not combined:
                        terms.pop(k)
                        terms.pop(j)
                        terms.pop(i)
                        found = True
                        break
                    continue

                    result = is_rank1_and_extract(combined)
                    if result is not None:
                        a_frac, b_frac, c_frac = result
                        _, a_int, b_int, c_int = scale_to_ints(a_frac, b_frac, c_frac)
                        terms.pop(k)
                        terms.pop(j)
                        terms.pop(i)
                        terms.append((a_int, b_int, c_int))
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
=======
    # Efficient merge: precompute sparse tensors and use overlap-based pairing
    def compute_sparse(terms_list):
        """Precompute sparse tensors for all terms."""
        return [term_to_sparse(*t) for t in terms_list]
    
    def try_merge_pair(sparse1, sparse2):
        """Try to merge two sparse tensors into a rank-1 term. Returns None or (a,b,c) dicts."""
        combined = dict(sparse1)
        for k, v in sparse2.items():
            combined[k] = combined.get(k, 0) + v
        combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}
        if not combined:
            return 'cancel'
        result = is_rank1_and_extract(combined)
        if result is not None:
            a_frac, b_frac, c_frac = result
            _, a_int, b_int, c_int = scale_to_ints(a_frac, b_frac, c_frac)
            return (a_int, b_int, c_int)
        return None
    
    def try_merge_triple(sparse1, sparse2, sparse3):
        """Try to merge three sparse tensors."""
        combined = {}
        for tt in [sparse1, sparse2, sparse3]:
            for k, v in tt.items():
                combined[k] = combined.get(k, 0) + v
        combined = {k: v for k, v in combined.items() if abs(v) > 1e-12}
        if not combined:
            return 'cancel'
        result = is_rank1_and_extract(combined)
        if result is not None:
            a_frac, b_frac, c_frac = result
            _, a_int, b_int, c_int = scale_to_ints(a_frac, b_frac, c_frac)
            return (a_int, b_int, c_int)
        return None
    
    # Iterative pairwise merging with precomputed sparses
    for pass_num in range(50):
        if time.time() > deadline - 10:
            break
        sparses = compute_sparse(terms)
        merged_any = False
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            n_terms = len(terms)
            for i in range(n_terms):
                if time.time() > deadline - 5:
                    break
                if i >= len(terms):
                    continue
                for j in range(i + 1, len(terms)):
                    if time.time() > deadline - 5:
                        break
                    result = try_merge_pair(sparses[i], sparses[j])
                    if result is not None:
                        if result == 'cancel':
                            terms.pop(j)
                            terms.pop(i)
                            sparses.pop(j)
                            sparses.pop(i)
                        else:
                            terms[i] = result
                            sparses[i] = term_to_sparse(*result)
                            terms.pop(j)
                            sparses.pop(j)
                        merged_any = True
                        changed = True
                        break
                if changed:
                    break
        if not merged_any:
            break
    
    # Triple merges with precomputed sparses
    for pass_num in range(20):
        if time.time() > deadline - 5:
            break
        sparses = compute_sparse(terms)
        found = False
        n_terms = len(terms)
        for i in range(n_terms):
            if time.time() > deadline - 3:
                break
            if i >= len(terms):
                continue
            for j in range(i + 1, len(terms)):
                if time.time() > deadline - 3:
                    break
                if j >= len(terms):
                    continue
                for k in range(j + 1, len(terms)):
                    if time.time() > deadline - 3:
                        break
                    if k >= len(terms):
                        continue
                    result = try_merge_triple(sparses[i], sparses[j], sparses[k])
                    if result is not None:
                        if result == 'cancel':
                            terms.pop(k)
                            terms.pop(j)
                            terms.pop(i)
                            sparses.pop(k)
                            sparses.pop(j)
                            sparses.pop(i)
                        else:
                            terms[i] = result
                            sparses[i] = term_to_sparse(*result)
                            terms.pop(k)
                            terms.pop(j)
                            sparses.pop(k)
                            sparses.pop(j)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
>>>>>>> REPLACE

Now let me also improve the `is_rank1_and_extract` function to be more robust:

