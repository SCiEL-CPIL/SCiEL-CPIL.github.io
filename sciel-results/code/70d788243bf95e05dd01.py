        idx = 0
        while idx < len(comps) and (time.time() - start_time) < DEADLINE - 2:
            # Try removing idx
            new_comps = comps[:idx] + comps[idx+1:]
            if _is_sorting_network_packed(new_comps, inputs):
                comps = new_comps
                # Do NOT increment idx, because the next candidate is now at idx
            else:
                idx += 1
        