import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    def calc_scores(A):
        M = A @ A
        masked = np.where(A > 0, M, 0)
        max_b = int(masked.max())
        count_max = int(np.sum(masked == max_b)) // 2
        sum_books = int(np.sum(masked)) // 2
        return max_b, count_max, sum_books, M, masked

    # Initialize with a random graph
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    initial_edges = all_edges[:target_edges]
    
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in initial_edges:
        A[u, v] = 1
        A[v, u] = 1
        
    best_score = calc_scores(A)[:3]
    best_A = A.copy()
    
    stuck_counter = 0
    current_score = best_score
    
    while time.time() < deadline:
        # Evaluate current
        cur_max, cur_cnt, cur_sum, M, masked = calc_scores(A)
        current_score = (cur_max, cur_cnt, cur_sum)
        
        # Update best
        if (cur_max < best_score[0]) or \
           (cur_max == best_score[0] and cur_cnt < best_score[1]) or \
           (cur_max == best_score[0] and cur_cnt == best_score[1] and cur_sum < best_score[2]):
            best_score = current_score
            best_A = A.copy()
            stuck_counter = 0
            
            if best_score[0] <= 7:
                break
        else:
            stuck_counter += 1

        # If stuck for a long time, perform a random kick (accept worse move)
        if stuck_counter > 100:
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
            non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
            if edges_list and non_edges_list:
                u, v = random.choice(edges_list)
                x, y = random.choice(non_edges_list)
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                stuck_counter = 0
                continue

        # Generate candidates
        max_idxs = np.argwhere(masked == cur_max)
        max_edges = [(u, v) for u, v in max_idxs if u < v]
        
        if not max_edges:
            break
            
        u, v = random.choice(max_edges)
        
        best_swap_score = None
        best_swap = None
        
        # Sample 30 random non-edges
        for _ in range(30):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y or A[x, y] == 1:
                continue
            if (x, y) == (u, v) or (x, y) == (v, u):
                continue
                
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            new_max, new_cnt, new_sum, _, _ = calc_scores(A)
            new_score = (new_max, new_cnt, new_sum)
            
            if best_swap_score is None or \
               (new_score[0] < best_swap_score[0]) or \
               (new_score[0] == best_swap_score[0] and new_score[1] < best_swap_score[1]) or \
               (new_score[0] == best_swap_score[0] and new_score[1] == best_swap_score[1] and new_score[2] < best_swap_score[2]):
                best_swap_score = new_score
                best_swap = (x, y)
                
            A[u, v] = 1
            A[v, u] = 1
            A[x, y] = 0
            A[y, x] = 0
            
        if best_swap is not None:
            x, y = best_swap
            if (best_swap_score[0] < cur_max) or \
               (best_swap_score[0] == cur_max and best_swap_score[1] < cur_cnt) or \
               (best_swap_score[0] == cur_max and best_swap_score[1] == cur_cnt and best_swap_score[2] < cur_sum):
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    if len(edges) != target_edges:
        if len(edges) > target_edges:
            edges = edges[:target_edges]
        else:
            existing = set(tuple(e) for e in edges)
            for u, v in all_edges:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))

    return edges
