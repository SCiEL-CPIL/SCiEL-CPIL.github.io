import numpy as np

def construct():
    # 8-point Gauss-Legendre quadrature on [0, 1]
    x, w = np.polynomial.legendre.leggauss(8)
    nodes = (x + 1.0) / 2.0
    weights = w / 2.0
    
    # Ensure strict increasing order and positive weights
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    
    # Verify constraints
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(np.diff(nodes) > 0)
    assert np.all(weights > 0)
    assert abs(np.sum(weights) - 1.0) < 1e-12
    
    return {"nodes": nodes, "weights": weights}
