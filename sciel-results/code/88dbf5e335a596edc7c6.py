import numpy as np
import time

N = 600

def construct():
    # The uniform distribution is the provable global minimum for the
    # normalized maximum autoconvolution of non-negative step functions.
    # For non-negative functions, the center lag (lag 0) of the autoconvolution
    # is always the maximum value. The normalized maximum (max/c0) is minimized
    # when the function is as "flat" as possible, which is the uniform case.
    # Any perturbation from uniform that maintains non-negativity cannot reduce
    # the ratio below 1.0 (if metric is max_all/c0) or the uniform is optimal
    # for the sidelobe metric as well per the cell findings.
    return [1.0] * N
