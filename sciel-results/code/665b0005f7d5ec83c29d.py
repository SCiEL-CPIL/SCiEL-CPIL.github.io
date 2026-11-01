<<<<<<< SEARCH
    final_params = best_params
    inner, oc, os, oa = parse_params(final_params)
    
    # Final validation and fallback
    if not is_valid(inner, oc, os, oa):
        # Use the best initial config as fallback
        config, oc_init, os_init, oa_init = initial_configs[0]
        inner = [(p[0], p[1], p[2]) for p in config]
        oc = tuple(oc_init)
        os = os_init
        oa = oa_init
        
        # Expand if needed
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.1
            if os > 10:
                break

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
=======
    final_params = best_params
    inner, oc, os, oa = parse_params(final_params)
    
    # Final validation and fallback
    if not is_valid(inner, oc, os, oa):
        # Use the best initial config as fallback
        config, oc_init, oa_init = initial_configs[0]
        inner = [(p[0], p[1], p[2]) for p in config]
        oc = tuple(oc_init)
        os = compute_minimal_outer(inner, oc, oa_init)
        oa = oa_init
        
        # Expand if needed
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.1
            if os > 10:
                break

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
