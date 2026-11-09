import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_size = 0
    best_set = []
    
    # Greedy initialization
    greedy_set = []
    for x in range(N, 0, -1):
        if _can_add_desc(greedy_set, x):
            greedy_set.append(x)
            
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = list(greedy_set)
        
    candidates = list(range(N, 0, -1))
    
    def _can_add_desc(S, x):
        """
        S contains elements all > x.
        Check if x can be added.
        Constraint: For all b, c in S with b <= c, x does not divide b + c.
        """
        m = len(S)
        for i in range(m):
            b = S[i]
            for j in range(i, m):
                c = S[j]
                if (b + c) % x == 0:
                    return False
        return True

    def search(idx, current_set, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if current_size + (N - idx) <= best_size:
            return
            
        if idx == N:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        x = candidates[idx]
        
        # Option 1: Include x
        if _can_add_desc(current_set, x):
            current_set.append(x)
            search(idx + 1, current_set, current_size + 1)
            current_set.pop()
            
        # Option 2: Exclude x
        search(idx + 1, current_set, current_size)

    search(0, [], 0)
    
    return sorted(best_set)

if __name__ == "__main__":
    _s = construct()
    def _validate(S):
        L = sorted(S)
        m = len(L)
        for i in range(m):
            a = L[i]
            for j in range(i + 1, m):
                b = L[j]
                for k in range(j, m):
                    c = L[k]
                    if (b + c) % a == 0:
                        return False
        return True
        
    assert _validate(_s), "constructed set is invalid"
    print("size", len(_s))
