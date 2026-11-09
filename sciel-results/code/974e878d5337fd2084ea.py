"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Greedy Bottleneck Elimination with composite urgency scoring.
    
    Strategy: Prioritize operations that are:
    1. Long (high min processing time) - LPT to avoid bottlenecks
    2. Have high remaining work - more total work left
    3. Have few remaining ops - closer to completion (urgency)
    4. Have low machine flexibility - constrained jobs need priority
    
    We use a weighted composite score to balance these signals.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    flexibilities = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    
    # Normalize to [0, 1] range
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_min_dur = normalize(min_durations)       # Higher = longer min duration = more bottleneck-prone
    norm_work = normalize(remaining_works)        # Higher = more work remaining
    norm_ops = normalize(remaining_ops)           # Higher = more ops remaining (less urgent)
    norm_flex = normalize(flexibilities)          # Higher = more flexible (less constrained)
    
    # Composite score:
    # - High weight on min duration (LPT bottleneck elimination)
    # - Moderate weight on remaining work
    # - Small negative weight on remaining ops (prefer fewer remaining = more urgent)
    # - Small negative weight on flexibility (prefer less flexible = more constrained)
    
    w_min_dur = 0.40
    w_work = 0.30
    w_ops = -0.15
    w_flex = -0.15
    
    composite = (w_min_dur * norm_min_dur + 
                 w_work * norm_work + 
                 w_ops * norm_ops + 
                 w_flex * norm_flex)
    
    return int(np.argmax(composite))
# EVOLVE-BLOCK-END
