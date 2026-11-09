import math
import heapq

def get_sf(n):
    res = 1
    p = 2
    while p * p <= n:
        if n % p == 0:
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count % 2 == 1:
                res *= p
        p += 1
    if n > 1:
        res *= n
    return res

def construct():
    target = 210
    
    sf_cache = {}
    def get_sf_cached(n):
        if n not in sf_cache:
            sf_cache[n] = get_sf(n)
        return sf_cache[n]

    dist = {1: 210}
    parent = {1: None}
    pq = [(210, 1)]
    
    found_max = 420
    found_path = None
    
    max_x_search = 1000
    
    while pq:
        max_val, s = heapq.heappop(pq)
        
        if s == target:
            # Reconstruct path
            path = []
            cur = s
            while parent[cur] is not None:
                prev_s, x = parent[cur]
                path.append(x)
                cur = prev_s
                
            if len(path) <= 100:
                # Verify
                prod = 210
                for x in path:
                    prod *= x
                sqrt_prod = math.isqrt(prod)
                if sqrt_prod * sqrt_prod == prod:
                    found_max = max_val
                    found_path = path
                    break
            # If path too long, we might continue searching for another path?
            # Dijkstra guarantees minimal max value for the state.
            # If the minimal max value path is too long, is there a longer max value path that is shorter?
            # Possibly. But we want to minimize max value primarily.
            # The task says "Minimize max(selected integers)-210".
            # It does not say "then minimize count".
            # But count must be <= 100.
            # If the minimal max value solution has > 100 elements, it is invalid.
            # We should then look for the next best max value.
            # However, Dijkstra pops states in order of max value.
            # So the first time we reach target, it has the minimal max value.
            # If that path is invalid, we need to find the next minimal max value path that is valid.
            # This requires keeping multiple paths or restarting.
            # Given the small numbers, it's very unlikely to need > 100 elements.
            # 242, 420 has 2 elements.
            # So we can assume the first found is valid.
            pass
        
        if max_val >= found_max:
            continue
            
        for x in range(211, max_x_search + 1):
            sx = get_sf_cached(x)
            g = math.gcd(s, sx)
            new_s = (s * sx) // (g * g)
            
            new_max = max(max_val, x)
            
            if new_max < dist.get(new_s, float('inf')):
                dist[new_s] = new_max
                parent[new_s] = (s, x)
                heapq.heappush(pq, (new_max, new_s))
                
    if found_path:
        return found_path
    else:
        return [242, 420]
