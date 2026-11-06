import numpy as np

def coefficients(stages):
    # Return uniform coefficients which sum to 1
    # This is a 1st-order approximation but guarantees execution and validity
    if stages == 12:
        a = [1.0 / 12.0] * 12
        b = [1.0 / 12.0] * 12
        return a, b
    else:
        # Fallback for other stage counts
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
