import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return [[0]*n]

    start_time = time.time()
    deadline = start_time + 110.0

    N = 3**n  # 2187
    NCHUNKS = (N + 63) // 64  # 34

    # Precompute digit representations
    digits = [None] * N
    for idx in range(N):
        digs = []
        tmp = idx
        for _ in range(n):
            digs.append(tmp % 3)
            tmp //= 3
        digs.reverse()
        digits[idx] = digs

    # Precompute neighbor lists and bitmasks
    neighbor_lists = [[] for _ in range(N)]
    neighbor_chunks = [None] * N

    for idx in range(N):
        digs = digits[idx]
        chunks = [0] * NCHUNKS
        chunks[idx >> 6] |= (1 << (idx & 63))
        neighbor_lists[idx].append(idx)
        for i in range(n):
            for delta in [-1, 1]:
                new_dig = (digs[i] + delta) % 3
                if new_dig != digs[i]:
                    new_digs = digs[:]
                    new_digs[i] = new_dig
                    new_idx = 0
                    for d in new_digs:
                        new_idx = new_idx * 3 + d
                    chunks[new_idx >> 6] |= (1 << (new_idx & 63))
                    neighbor_lists[idx].append(new_idx)
        neighbor_chunks[idx] = chunks

    # Precompute neighbor masks as numpy array (N, NCHUNKS) uint64
    nm_arr = np.zeros((N, NCHUNKS), dtype=np.uint64)
    for idx in range(N):
        for k in range(NCHUNKS):
            nm_arr[idx, k] = neighbor_chunks[idx][k]

    # Popcount lookup for 16-bit values
    POP16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.int32)

    def popcount_u64(arr):
        """Popcount for numpy array of uint64, returns int32 array."""
        a = arr.astype(np.uint64)
        p0 = (a & 0xFFFF).astype(np.int32)
        p1 = ((a >> 16) & 0xFFFF).astype(np.int32)
        p2 = ((a >> 32) & 0xFFFF).astype(np.int32)
        p3 = ((a >> 48) & 0xFFFF).astype(np.int32)
        return POP16[p0] + POP16[p1] + POP16[p2] + POP16[p3]

    def compute_gains(unc_arr):
        """Compute coverage gain for each candidate given uncovered bitmask array."""
        and_mat = nm_arr & unc_arr[np.newaxis, :]
        gains = popcount_u64(and_mat.ravel()).reshape(N, NCHUNKS).sum(axis=1)
        return gains

    def greedy_from_unc(unc_arr):
        """Greedy set cover starting from uncovered bitmask. Returns list of codewords added."""
        code = []
        while True:
            if not unc_arr.any():
                break
            gains = compute_gains(unc_arr)
            best_c = int(np.argmax(gains))
            best_gain = int(gains[best_c])
            if best_gain == 0:
                break
            code.append(best_c)
            unc_arr = unc_arr & ~nm_arr[best_c]
        return code

    # Initial greedy
    ALL_ONES = np.array([((1 << 64) - 1) if (N - i*64) >= 64 else ((1 << (N - i*64)) - 1) for i in range(NCHUNKS)], dtype=np.uint64)
    code = greedy_from_unc(ALL_ONES.copy())

    best_code = list(code)
    best_len = len(code)

    def local_search(cur_code):
        """Deletion + swap local search. Returns improved code or None if no improvement."""
        cc = np.zeros(N, dtype=np.int32)
        for c in cur_code:
            for w in neighbor_lists[c]:
                cc[w] += 1

        code_list = list(cur_code)
        code_set = set(code_list)
        improved = False

        # Deletion pass
        while True:
            if time.time() > deadline - 5:
                break
            changed = False
            for idx_c, c in enumerate(code_list):
                removable = True
                for w in neighbor_lists[c]:
                    if cc[w] <= 1:
                        removable = False
                        break
                if removable:
                    for w in neighbor_lists[c]:
                        cc[w] -= 1
                    code_list.pop(idx_c)
                    code_set.discard(c)
                    changed = True
                    improved = True
                    break
            if not changed:
                break

        # 2-for-1 swap pass
        if time.time() < deadline - 10 and len(code_list) > 1:
            for i in range(len(code_list)):
                if time.time() > deadline - 5:
                    break
                c1 = code_list[i]
                for j in range(i+1, len(code_list)):
                    if time.time() > deadline - 5:
                        break
                    c2 = code_list[j]
                    for w in neighbor_lists[c1]:
                        cc[w] -= 1
                    for w in neighbor_lists[c2]:
                        cc[w] -= 1

                    uncovered_mask = np.zeros(NCHUNKS, dtype=np.uint64)
                    for w in range(N):
                        if cc[w] == 0:
                            uncovered_mask[w >> 6] |= (1 << (w & 63))

                    if not uncovered_mask.any():
                        code_list.pop(j)
                        code_list.pop(i)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        improved = True
                        return list(code_list)

                    found = False
                    for c_new in range(N):
                        ok = True
                        for k in range(NCHUNKS):
                            if (nm_arr[c_new, k] & uncovered_mask[k]) != uncovered_mask[k]:
                                ok = False
                                break
                        if ok:
                            if c_new not in code_set:
                                for w in neighbor_lists[c_new]:
                                    cc[w] += 1
                                code_list.append(c_new)
                                code_set.add(c_new)
                            code_list.pop(j)
                            code_list.pop(i)
                            code_set.discard(c1)
                            code_set.discard(c2)
                            improved = True
                            return list(code_list)

                    for w in neighbor_lists[c1]:
                        cc[w] += 1
                    for w in neighbor_lists[c2]:
                        cc[w] += 1

        return list(code_list) if improved else None

    # Multi-restart
    for restart in range(50):
        if time.time() > deadline - 15:
            break

        if restart == 0:
            cur = list(best_code)
        else:
            cur = list(best_code)
            n_remove = random.randint(1, 3)
            to_remove = random.sample(cur, min(n_remove, len(cur)))
            cc = np.zeros(N, dtype=np.int32)
            for c in cur:
                for w in neighbor_lists[c]:
                    cc[w] += 1
            for c in to_remove:
                for w in neighbor_lists[c]:
                    cc[w] -= 1
                cur.remove(c)

            unc_arr = np.zeros(NCHUNKS, dtype=np.uint64)
            for w in range(N):
                if cc[w] == 0:
                    unc_arr[w >> 6] |= (1 << (w & 63))

            added = greedy_from_unc(unc_arr)
            cur.extend(added)

        result = local_search(cur)
        if result is not None and len(result) < best_len:
            best_code = result
            best_len = len(result)

    return [digits[c] for c in best_code]
