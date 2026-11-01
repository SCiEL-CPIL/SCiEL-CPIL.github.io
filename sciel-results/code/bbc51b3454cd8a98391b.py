try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    HAS_MILP = True
except ImportError:
    HAS_MILP = False
