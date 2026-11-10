import time

def construct():
    deadline = time.time() + 80
    
    def is_valid(arr):
        n = len(arr)
        if n == 0:
            return True
        sums = [set() for _ in range(n+1)]
        sums[0].add(0)
        for x in arr:
            for k in range(n, 0, -1):
                if sums[k-1]:
                    sums[k].update(s + x for s in sums[k-1])
        seen = set()
        for k in range(n+1):
            for s in sums[k]:
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    result = []
    for x in range(256, 0, -1):
        if time.time() > deadline - 5:
            break
        if len(result) >= 60:
            break
        trial = result + [x]
        if len(trial) <= 20:
            if is_valid(trial):
                result = trial
        else:
            s = sum(result)
            if x > s:
                result = trial
    
    return result
