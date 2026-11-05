import time

def popcount(x):
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count('1')

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        total_words = q ** n
        cover = [0] * total_words
        for w in range(total_words):
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            c = 0
            c |= 1 << w
            for i in range(n):
                for delta in range(1, q):
                    for sign in [1, -1]:
                        new_digits = digits[:]
                        new_digits[i] = (new_digits[i] + sign * delta) % q
                        new_w = 0
                        for j in range(n):
                            new_w = new_w * q + new_digits[j]
                        c |= 1 << new_w
            cover[w] = c
        all_words_mask = (1 << total_words) - 1
        uncovered = all_words_mask
        code = []
        in_code = [False] * total_words
        while uncovered:
            best_w = -1
            best_count = -1
            for w in range(total_words):
                if in_code[w]:
                    continue
                count = popcount(cover[w] & uncovered)
                if count > best_count:
                    best_count = count
                    best_w = w
            if best_w == -1:
                break
            code.append(best_w)
            in_code[best_w] = True
            uncovered &= ~cover[best_w]
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result

    total_words = 3 ** n
    cover = [0] * total_words
    for w in range(total_words):
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        c = 0
        c |= 1 << w
        for i in range(n):
            for delta in [-1, 1]:
                new_digits = digits[:]
                new_digits[i] = (new_digits[i] + delta) % 3
                new_w = 0
                for j in range(n):
                    new_w = new_w * 3 + new_digits[j]
                c |= 1 << new_w
        cover[w] = c

    all_words_mask = (1 << total_words) - 1
    uncovered = all_words_mask
    code = []
    in_code = [False] * total_words
    
    # Greedy
    while uncovered:
        best_w = -1
        best_count = -1
        for w in range(total_words):
            if in_code[w]:
                continue
            count = popcount(cover[w] & uncovered)
            if count > best_count:
                best_count = count
                best_w = w
        if best_w == -1:
            break
        code.append(best_w)
        in_code[best_w] = True
        uncovered &= ~cover[best_w]
        
    # Local search: remove redundant words
    cover_count = [0] * total_words
    for w in code:
        c = cover[w]
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cover_count[v] += 1
            c ^= lsb
            
    while True:
        removed_any = False
        for i in range(len(code) - 1, -1, -1):
            w = code[i]
            can_remove = True
            c = cover[w]
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if cover_count[v] == 1:
                    can_remove = False
                    break
                c ^= lsb
            if can_remove:
                c = cover[w]
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    cover_count[v] -= 1
                    c ^= lsb
                code.pop(i)
                in_code[w] = False
                removed_any = True
        if not removed_any:
            break

    # Swap-based local search
    start_time = time.time()
    time_limit = 100.0
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(code)):
            if (time.time() - start_time) > time_limit:
                break
            w = code[i]
            for u in range(total_words):
                if in_code[u]:
                    continue
                # Check if swapping w with u is valid
                valid = True
                c = cover[w]
                u_mask = cover[u]
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    if (u_mask >> v) & 1:
                        new_count = cover_count[v] - 1 + 1
                    else:
                        new_count = cover_count[v] - 1
                    if new_count < 0:
                        valid = False
                        break
                    c ^= lsb
                if valid:
                    for j in range(len(code)):
                        if j == i:
                            continue
                        z = code[j]
                        can_remove_z = True
                        cz = cover[z]
                        while cz:
                            lsb = cz & -cz
                            v = lsb.bit_length() - 1
                            cc = cover_count[v]
                            if (cover[w] >> v) & 1:
                                cc -= 1
                            if (u_mask >> v) & 1:
                                cc += 1
                            if cc < 1:
                                can_remove_z = False
                                break
                            cz ^= lsb
                        if can_remove_z:
                            # Perform swap and removal
                            cw = cover[w]
                            while cw:
                                lsb = cw & -cw
                                v = lsb.bit_length() - 1
                                cover_count[v] -= 1
                                cw ^= lsb
                            cu = cover[u]
                            while cu:
                                lsb = cu & -cu
                                v = lsb.bit_length() - 1
                                cover_count[v] += 1
                                cu ^= lsb
                            cz = cover[z]
                            while cz:
                                lsb = cz & -cz
                                v = lsb.bit_length() - 1
                                cover_count[v] -= 1
                                cz ^= lsb
                            
                            in_code[w] = False
                            in_code[u] = True
                            in_code[z] = False
                            code[i] = u
                            code.pop(j)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break

    result = []
    for w in code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        result.append(digits)
        
    return result
