import math

def construct():
    selected = []
    
    # Look-ahead depth for tie-breaking
    LOOKAHEAD = 20
    
    for m in range(20, 201):
        valid_rs = []
        for r in range(m):
            valid = True
            for (m2, r2) in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                valid_rs.append(r)
        
        if not valid_rs:
            continue
            
        if len(valid_rs) == 1:
            selected.append((m, valid_rs[0]))
        else:
            best_r = None
            best_metric = -1
            
            for r in valid_rs:
                # Simulate next LOOKAHEAD moduli
                temp_selected = selected + [(m, r)]
                added_sum = 0.0
                added_count = 0
                
                # We only simulate a few steps ahead to keep it fast
                for m_prime in range(m + 1, min(m + 1 + LOOKAHEAD, 201)):
                    found = False
                    # To make simulation faster, we can just check if *any* r_prime exists
                    # We don't need to pick the best one for the simulation, just if it's possible.
                    # But picking one affects subsequent steps. We pick the first valid one.
                    for r_prime in range(m_prime):
                        valid_prime = True
                        for (m3, r3) in temp_selected:
                            g3 = math.gcd(m_prime, m3)
                            if r_prime % g3 == r3 % g3:
                                valid_prime = False
                                break
                        if valid_prime:
                            found = True
                            temp_selected.append((m_prime, r_prime))
                            added_sum += 1.0 / m_prime
                            added_count += 1
                            break
                    
                    if not found:
                        # If we can't add m_prime, we might still be able to add later ones?
                        # In a greedy simulation, if m_prime is blocked, we skip it.
                        # But for the metric, we just want to know how many we *can* add.
                        # Actually, if m_prime is blocked, we don't add it, so it doesn't block others.
                        # So we continue.
                        pass
                
                # Metric: Prefer more added moduli, then higher sum
                # Since 1/m is decreasing, maximizing count is a good proxy.
                # If counts are equal, maximize sum (which implies adding smaller m's).
                metric = added_count * 1000 + added_sum
                
                if metric > best_metric:
                    best_metric = metric
                    best_r = r
            
            selected.append((m, best_r))
            
    return [[m, r] for m, r in selected]
