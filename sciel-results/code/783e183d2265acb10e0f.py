import random
import time

def construct():
    n = 256
    best_A = []
    best_B = []
    best_score = 0
    start_time = time.time()
    time_limit = 55.0
    all_nums = list(range(1, n + 1))

    def is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
        for a in A:
            if (x + a) in sums_A:
                return False
            if (x - a) in diffs_B:
                return False
            if (a - x) in diffs_B:
                return False
        if (2 * x) in sums_A:
            return False
        return True

    def is_valid_add_B(B, sums_B, diffs_A, diffs_B, x):
        for b in B:
            if (x + b) in sums_B:
                return False
            if (x - b) in diffs_A:
                return False
            if (b - x) in diffs_A:
                return False
        if (2 * x) in sums_B:
            return False
        return True

    def update_add_A(A, sums_A, diffs_A, x):
        new_A = A | {x}
        new_sums_A = sums_A | {(x + a) for a in A} | {2 * x}
        new_diffs_A = diffs_A | {(x - a) for a in A} | {(a - x) for a in A}
        return new_A, new_sums_A, new_diffs_A

    def update_add_B(B, sums_B, diffs_B, x):
        new_B = B | {x}
        new_sums_B = sums_B | {(x + b) for b in B} | {2 * x}
        new_diffs_B = diffs_B | {(x - b) for b in B} | {(b - x) for b in B}
        return new_B, new_sums_B, new_diffs_B

    def recalc_sums_diffs(S):
        sums = set()
        diffs = set()
        s_list = list(S)
        for i in range(len(s_list)):
            s1 = s_list[i]
            for j in range(i, len(s_list)):
                s2 = s_list[j]
                sums.add(s1 + s2)
            for j in range(len(s_list)):
                if i != j:
                    s2 = s_list[j]
                    diffs.add(s1 - s2)
        return sums, diffs

    def greedy_init(seed):
        random.seed(seed)
        A = set()
        B = set()
        sums_A, diffs_A = set(), set()
        sums_B, diffs_B = set(), set()
        
        order = all_nums[:]
        random.shuffle(order)
        
        for x in order:
            if is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
                A, sums_A, diffs_A = update_add_A(A, sums_A, diffs_A, x)
            elif is_valid_add_B(B, sums_B, diffs_A, diffs_B, x):
                B, sums_B, diffs_B = update_add_B(B, sums_B, diffs_B, x)
                
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search(A, B, sums_A, diffs_A, sums_B, diffs_B, time_limit):
        current_score = len(A) + len(B)
        best_local_A, best_local_B = set(A), set(B)
        best_local_score = current_score
        
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() - start_time < time_limit:
            if no_improve_count > max_no_improve:
                break
            
            move_type = random.randint(0, 4)
            
            if move_type == 0: # Add to A
                candidates = [x for x in all_nums if x not in A and x not in B]
                if not candidates:
                    no_improve_count += 1
                    continue
                x = random.choice(candidates)
                if is_valid_add_A(A, sums_A, diffs_A, diffs_B, x):
                    A, sums_A, diffs_A = update_add_A(A, sums_A, diffs_A, x)
                    current_score += 1
                    no_improve_count = 0
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_A, best_local_B = set(A), set(B)
                else:
                    no_improve_count += 1
                    
            elif move_type == 1: # Add to B
                candidates = [x for x in all_nums if x not in A and x not in B]
                if not candidates:
                    no_improve_count += 1
                    continue
                x = random.choice(candidates)
                if is_valid_add_B(B, sums_B, diffs_A, diffs_B, x):
                    B, sums_B, diffs_B = update_add_B(B, sums_B, diffs_B, x)
                    current_score += 1
                    no_improve_count = 0
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_A, best_local_B = set(A), set(B)
                else:
                    no_improve_count += 1

            elif move_type == 2: # Swap in A
                if not A:
                    no_improve_count += 1
                    continue
                x = random.choice(list(A))
                candidates = [y for y in all_nums if y not in A and y not in B]
                if not candidates:
                    no_improve_count += 1
                    continue
                
                temp_A = A - {x}
                temp_sums_A, temp_diffs_A = recalc_sums_diffs(temp_A)
                
                found_swap = False
                random.shuffle(candidates)
                for y in candidates:
                    if is_valid_add_A(temp_A, temp_sums_A, temp_diffs_A, diffs_B, y):
                        A, sums_A, diffs_A = update_add_A(temp_A, temp_sums_A, temp_diffs_A, y)
                        found_swap = True
                        no_improve_count = 0
                        break
                
                if not found_swap:
                    no_improve_count += 1
                
            elif move_type == 3: # Swap in B
                if not B:
                    no_improve_count += 1
                    continue
                x = random.choice(list(B))
                candidates = [y for y in all_nums if y not in A and y not in B]
                if not candidates:
                    no_improve_count += 1
                    continue
                
                temp_B = B - {x}
                temp_sums_B, temp_diffs_B = recalc_sums_diffs(temp_B)
                
                found_swap = False
                random.shuffle(candidates)
                for y in candidates:
                    if is_valid_add_B(temp_B, temp_sums_B, diffs_A, temp_diffs_B, y):
                        B, sums_B, diffs_B = update_add_B(temp_B, temp_sums_B, temp_diffs_B, y)
                        found_swap = True
                        no_improve_count = 0
                        break
                
                if not found_swap:
                    no_improve_count += 1

            elif move_type == 4: # Move A to B
                if not A:
                    no_improve_count += 1
                    continue
                x = random.choice(list(A))
                
                temp_A = A - {x}
                temp_sums_A, temp_diffs_A = recalc_sums_diffs(temp_A)
                
                if is_valid_add_B(B, sums_B, temp_diffs_A, diffs_B, x):
                    B, sums_B, diffs_B = update_add_B(B, sums_B, diffs_B, x)
                    A, sums_A, diffs_A = temp_A, temp_sums_A, temp_diffs_A
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if time.time() - start_time > time_limit:
                break

        return best_local_A, best_local_B, best_local_score

    max_restarts = 30
    for i in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        A, B, sums_A, diffs_A, sums_B, diffs_B = greedy_init(i)
        
        local_A, local_B, local_score = local_search(A, B, sums_A, diffs_A, sums_B, diffs_B, time_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_A, best_B = list(local_A), list(local_B)
            
        if best_score >= 36:
            break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
