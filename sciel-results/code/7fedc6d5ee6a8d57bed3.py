import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    def standard_superperm_string(k):
        if k == 1:
            return "1"
        if k == 2:
            return "121"
        prev_str = standard_superperm_string(k - 1)
        prev_list = [int(c) for c in prev_str]
        
        perms = []
        seen = set()
        for i in range(len(prev_list) - (k - 2)):
            window = tuple(prev_list[i:i + k - 1])
            if len(set(window)) == k - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_list = []
        for p in perms:
            new_list.append(k)
            new_list.extend(p)
            
        return "".join(map(str, new_list))

    prev_str = standard_superperm_string(n - 1)
    prev_list = [int(c) for c in prev_str]
    blocks = []
    seen = set()
    for i in range(len(prev_list) - (n - 2)):
        window = tuple(prev_list[i:i + n - 1])
        if len(set(window)) == n - 1:
            if window not in seen:
                seen.add(window)
                blocks.append(window)
                
    num_blocks = len(blocks)
    block_index = {b: i for i, b in enumerate(blocks)}
    
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = min(len(bi), len(bj))
            ov = 0
            for k in range(max_k, -1, -1):
                if bi[-k:] == bj[:k]:
                    ov = k
                    break
            overlap[i][j] = ov

    order = list(range(num_blocks))
    
    def calc_len(order):
        total = n - 1
        for i in range(len(order) - 1):
            total += (n - 1) - overlap[order[i]][order[i+1]]
        return total

    best_len = calc_len(order)
    
    start_time = time.time()
    time_limit = 110
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 2-opt
        for i in range(num_blocks - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, num_blocks):
                if time.time() - start_time > time_limit:
                    break
                
                delta = 0
                if i > 0:
                    delta -= overlap[order[i-1]][order[i]]
                for k in range(i, j):
                    delta -= overlap[order[k]][order[k+1]]
                if j < num_blocks - 1:
                    delta -= overlap[order[j]][order[j+1]]
                
                if i > 0:
                    delta += overlap[order[i-1]][order[j]]
                for k in range(i, j):
                    delta += overlap[order[k+1]][order[k]]
                if j < num_blocks - 1:
                    delta += overlap[order[i]][order[j+1]]
                
                new_len = best_len - delta
                if new_len < best_len:
                    seg = order[i:j+1]
                    seg.reverse()
                    order[i:j+1] = seg
                    best_len = new_len
                    improved = True

    word = list(blocks[order[0]])
    for i in range(1, len(order)):
        prev_idx = order[i-1]
        curr_idx = order[i]
        ov = overlap[prev_idx][curr_idx]
        word.extend(blocks[curr_idx][ov:])
        
    return "".join(map(str, word))
