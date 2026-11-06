import random
import time

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)

    def is_k33_free_add(u, v, M):
        A = M[v] & ~(1 << u)
        B = M[u] & ~(1 << v)

        # Quick reject: if either set has < 2 elements, no K2,2 possible
        if A & (A - 1) == 0:
            return True
        if B & (B - 1) == 0:
            return True

        a_count = bin(A).count('1')
        b_count = bin(B).count('1')

        if a_count <= b_count:
            a_list = []
            temp = A
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                a_list.append(idx)
                temp ^= lsb
            for i in range(len(a_list)):
                for j in range(i + 1, len(a_list)):
                    common = M[a_list[i]] & M[a_list[j]] & B
                    if common & (common - 1):
                        return False
        else:
            b_list = []
            temp = B
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                b_list.append(idx)
                temp ^= lsb
            for i in range(len(b_list)):
                for j in range(i + 1, len(b_list)):
                    common = M[b_list[i]] & M[b_list[j]] & A
                    if common & (common - 1):
                        return False
        return True

    def greedy_fill(M, order):
        for u, v in order:
            if is_k33_free_add(u, v, M):
                M[u] |= (1 << v)
                M[v] |= (1 << u)
        return sum(bin(M[i]).count("1") for i in range(n)) // 2

    def local_search(M, deadline):
        count = sum(bin(M[i]).count("1") for i in range(n)) // 2
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_edges = set()
            for i in range(n):
                temp = M[i]
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    if i < j:
                        cur_edges.add((i, j))
                    temp ^= lsb
            for rem_edge in list(cur_edges):
                u, v = rem_edge
                M[u] &= ~(1 << v)
                M[v] &= ~(1 << u)
                added_edges = []
                for cu, cv in all_edges:
                    if (cu, cv) == rem_edge:
                        continue
                    if M[cu] & (1 << cv):
                        continue
                    if is_k33_free_add(cu, cv, M):
                        M[cu] |= (1 << cv)
                        M[cv] |= (1 << cu)
                        added_edges.append((cu, cv))
                        if len(added_edges) >= 2:
                            break
                new_count = sum(bin(M[i]).count("1") for i in range(n)) // 2
                if new_count > count:
                    count = new_count
                    improved = True
                    break
                else:
                    for cu, cv in added_edges:
                        M[cu] &= ~(1 << cv)
                        M[cv] &= ~(1 << cu)
                    M[u] |= (1 << v)
                    M[v] |= (1 << u)
        return count

    def kick_and_fill(deadline):
        M = [0] * n
        kick_size = random.randint(5, 15)
        kick_edges = random.sample(all_edges, kick_size)
        for u, v in kick_edges:
            if is_k33_free_add(u, v, M):
                M[u] |= (1 << v)
                M[v] |= (1 << u)
        order = all_edges[:]
        random.shuffle(order)
        count = greedy_fill(M, order)
        count = local_search(M, deadline)
        return M, count

    def greedy_random(deadline):
        M = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        count = greedy_fill(M, order)
        return M, count

    best_M = [0] * n
    best_count = 0

    start_time = time.time()
    deadline = start_time + 115.0

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        if time.time() > deadline:
            break
        if iteration % 3 == 0:
            M, count = kick_and_fill(deadline)
        else:
            M, count = greedy_random(deadline)
        if count > best_count:
            best_count = count
            best_M = M[:]

    result = []
    for i in range(n):
        temp = best_M[i]
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            if i < j:
                result.append([i, j])
            temp ^= lsb
    return result
