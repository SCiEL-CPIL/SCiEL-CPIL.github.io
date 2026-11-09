import random

def construct():
    n = 128
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
            
    def is_valid_ap(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) >= 3

    def count_violations(colors):
        return sum(1 for ap in aps if not is_valid_ap(colors, ap))

    def solve_with_k_colors(k, num_restarts=50, max_steps=5000):
        for r in range(num_restarts):
            random.seed(12345 + r)
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Initial state
            ap_violated = [not is_valid_ap(colors, ap) for ap in aps]
            total_viol = sum(ap_violated)
            
            for step in range(max_steps):
                if total_viol == 0:
                    return colors
                
                # Pick a random position
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                
                best_color = old_color
                best_viol = total_viol
                
                for c in range(k):
                    if c == old_color:
                        continue
                    colors[pos] = c
                    
                    # Update violations for APs containing pos
                    new_viol = total_viol
                    for ap_idx in pos_to_aps[pos]:
                        was_violated = ap_violated[ap_idx]
                        is_now_violated = not is_valid_ap(colors, aps[ap_idx])
                        
                        if was_violated and not is_now_violated:
                            new_viol -= 1
                        elif not was_violated and is_now_violated:
                            new_viol += 1
                        # Update state for next iteration if we accept this move
                        # But we are just evaluating, so we need to track temporary changes?
                        # No, we can just calculate delta.
                        
                        # Actually, it's easier to just recompute the status of these specific APs
                        # and keep track of the delta.
                        
                        # Let's refine:
                        # We need to know the change in total_viol.
                        # Let's just calculate the new total_viol for this candidate move.
                        # This requires knowing the old state of all affected APs.
                        # We have `ap_violated` which reflects the CURRENT state (with old_color).
                        # So for each affected AP:
                        #   old_status = ap_violated[ap_idx]
                        #   new_status = not is_valid_ap(...)
                        #   delta += (1 if new_status else 0) - (1 if old_status else 0)
                        
                    # Recalculating delta properly:
                    new_total = total_viol
                    for ap_idx in pos_to_aps[pos]:
                        old_status = ap_violated[ap_idx]
                        new_status = not is_valid_ap(colors, aps[ap_idx])
                        if old_status == new_status:
                            continue
                        if new_status:
                            new_total += 1
                        else:
                            new_total -= 1
                    
                    if new_total < best_viol:
                        best_viol = new_total
                        best_color = c
                    
                    colors[pos] = old_color # Restore for next candidate
                
                # Apply best move or random move
                if best_color != old_color:
                    colors[pos] = best_color
                    # Update ap_violated and total_viol
                    for ap_idx in pos_to_aps[pos]:
                        old_status = ap_violated[ap_idx]
                        new_status = not is_valid_ap(colors, aps[ap_idx])
                        if old_status != new_status:
                            ap_violated[ap_idx] = new_status
                            if new_status:
                                total_viol += 1
                            else:
                                total_viol -= 1
                else:
                    # Random move to escape
                    new_c = random.choice([c for c in range(k) if c != old_color])
                    colors[pos] = new_c
                    for ap_idx in pos_to_aps[pos]:
                        old_status = ap_violated[ap_idx]
                        new_status = not is_valid_ap(colors, aps[ap_idx])
                        if old_status != new_status:
                            ap_violated[ap_idx] = new_status
                            if new_status:
                                total_viol += 1
                            else:
                                total_viol -= 1

        return None

    # Try 3 colors
    res = solve_with_k_colors(3, num_restarts=20, max_steps=5000)
    if res:
        return res
        
    # Try 4 colors
    res = solve_with_k_colors(4, num_restarts=20, max_steps=5000)
    if res:
        return res

    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

