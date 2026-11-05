    def run_greedy_and_remove():
        candidates = list(range(total))
        random.shuffle(candidates)
        
        uncovered_bits = (1 << total) - 1
        chosen = []
        chosen_set = set()
        
        while uncovered_bits:
            best_c = -1
            best_count = 0
            for c in candidates:
                if c in chosen_set:
                    continue
                inter = uncovered_bits & cov_bits[c]
                if inter == 0:
                    continue
                try:
                    count = inter.bit_count()
                except AttributeError:
                    count = bin(inter).count('1')
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1:
                break
            chosen.append(best_c)
            chosen_set.add(best_c)
            uncovered_bits &= ~cov_bits[best_c]
            
        # Removal
        cover_counts = np.zeros(total, dtype=int)
        for c in chosen:
            for w in cov[c]:
                cover_counts[w] += 1
        removed_any = True
        while removed_any and time.time() < deadline - 2:
            removed_any = False
            for i in range(len(chosen)):
                if time.time() > deadline - 2:
                    break
                c = chosen[i]
                can_remove = True
                for w in cov[c]:
                    if cover_counts[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for w in cov[c]:
                        cover_counts[w] -= 1
                    chosen.pop(i)
                    chosen_set.remove(c)
                    removed_any = True
                    break
        return chosen

    best_code = run_greedy_and_remove()
    best_size = len(best_code)
    
    for i in range(50):
        if time.time() > deadline - 5:
            break
        current = run_greedy_and_remove()
        if len(current) < best_size:
            best_code = current
            best_size = len(current)
            
    result = [int_to_digits(c, n, q) for c in best_code]
    return result
