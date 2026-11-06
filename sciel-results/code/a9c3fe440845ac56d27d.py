import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    def can_add(x, S_list):
        """Check if x can be added to sorted list S_list maintaining validity."""
        for a in S_list:
            if a >= x:
                continue
            # a < x: check constraints involving x
            # b=x, c=x: a ∤ 2x
            if (2 * x) % a == 0:
                return False
            # b=x, c in S_list, c > x: a ∤ (x+c)
            for c in S_list:
                if c > x and (x + c) % a == 0:
                    return False
            # c=x, b in S_list, a < b < x: a ∤ (b+x)
            for b in S_list:
                if a < b < x and (b + x) % a == 0:
                    return False
        # a=x: b,c in S_list, b > x, c >= b: x ∤ (b+c)
        for i in range(len(S_list)):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, len(S_list)):
                c = S_list[j]
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        return True
    
    best_result = []
    best_size = 0
    
    # Try multiple starting lower bounds
    for L in range(n, max(1, n - 200), -1):
        if time.time() > deadline:
            break
        
        S_list = list(range(L, n + 1))
        
        candidates = list(range(1, L))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S_list):
                S_list.append(x)
                S_list.sort()
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_result = S_list[:]
    
    # Local search
    if best_result:
        current = best_result[:]
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                temp = [x for x in current if x != removed]
                for x in range(1, n + 1):
                    if x in temp:
                        continue
                    if can_add(x, temp):
                        new_set = temp + [x]
                        new_set.sort()
                        if len(new_set) > len(current):
                            current = new_set
                            improved = True
                            break
                if improved:
                    break
            if len(current) > best_size:
                best_size = len(current)
                best_result = current[:]
    
    return best_result if best_result else list(range(129, 257))

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
