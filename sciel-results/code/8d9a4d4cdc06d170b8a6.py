import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    K = 4
    colors = list(range(K))
    
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
            
    def count_violations(config):
        viol = 0
        for a, b, c, e in aps:
            ca = config[a]
            cb = config[b]
            cc = config[c]
            ce = config[e]
            # Count distinct
            d_count = 1
            if cb != ca: d_count += 1
            if cc != ca and cc != cb: d_count += 1
            if ce != ca and ce != cb and ce != cc: d_count += 1
            if d_count < 3:
                viol += 1
        return viol

    def local_search(initial_config, time_limit):
        config = initial_config[:]
        current_viol = count_violations(config)
        best_local_viol = current_viol
        best_local_config = config[:]
        
        while time.time() < time_limit:
            idx = random.randint(0, N - 1)
            old_color = config[idx]
            # Pick new color
            new_color = random.choice([c for c in colors if c != old_color])
            config[idx] = new_color
            
            new_viol = count_violations(config)
            
            if new_viol < current_viol:
                current_viol = new_viol
                if current_viol < best_local_viol:
                    best_local_viol = current_viol
                    best_local_config = config[:]
                    if best_local_viol == 0:
                        return best_local_config, 0
            else:
                config[idx] = old_color
                
        return best_local_config, best_local_viol

    best_penalty = float('inf')
    best_config = None
    
    for i in range(200):
        if time.time() > deadline - 5.0:
            break
        config = [random.choice(colors) for _ in range(N)]
        slice_time = time.time() + 0.2
        cfg, viol = local_search(config, slice_time)
        
        if viol < best_penalty:
            best_penalty = viol
            best_config = cfg
            
        if best_penalty == 0:
            return best_config

    K = 5
    colors = list(range(K))
    best_penalty_5 = float('inf')
    best_config_5 = None
    
    for i in range(100):
        if time.time() > deadline - 2.0:
            break
        config = [random.choice(colors) for _ in range(N)]
        slice_time = time.time() + 0.2
        cfg, viol = local_search(config, slice_time)
        
        if viol < best_penalty_5:
            best_penalty_5 = viol
            best_config_5 = cfg
            
        if best_penalty_5 == 0:
            return best_config_5

    if best_config is not None and best_penalty == 0:
        return best_config
    if best_config_5 is not None and best_penalty_5 == 0:
        return best_config_5
    
    # Fallback to greedy
    colors_greedy = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors_greedy[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors_greedy[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors_greedy.append(color);break
    return colors_greedy
