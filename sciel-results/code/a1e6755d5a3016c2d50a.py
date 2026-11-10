import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
      n = 16
      start_time = time.time()
      deadline = start_time + 590

      def get_score(points):
          dists = pdist(points, 'euclidean')
          d_min = np.min(dists)
          d_max = np.max(dists)
          if d_max < 1e-10: return 0.0
          return (d_min / d_max) ** 2

      def soft_min(dists, beta=50.0):
          """Smooth approximation of min using log-sum-exp"""
          if len(dists) == 0:
              return 0.0
          min_d = np.min(dists)
          return -min_d - (1.0/beta) * np.log(np.sum(np.exp(-beta * (dists - min_d))))

      def soft_max(dists, beta=50.0):
          """Smooth approximation of max using log-sum-exp"""
          if len(dists) == 0:
              return 0.0
          max_d = np.max(dists)
          return max_d + (1.0/beta) * np.log(np.sum(np.exp(beta * (dists - max_d))))

      def objective_scaled(params, beta=50.0):
          """
          Objective with radial scaling: fix d_max=1, maximize soft-min distance.
          Returns negative soft-min to minimize.
          """
          pts = params.reshape((n, 2))
          dists = pdist(pts, 'euclidean')
          
          d_max = np.max(dists)
          if d_max < 1e-10:
              return 1.0  # Bad configuration
              
          # Scale points so d_max = 1
          scaled_pts = pts / d_max
          scaled_dists = pdist(scaled_pts, 'euclidean')
          
          # Maximize soft-min of scaled distances
          sm = soft_min(scaled_dists, beta)
          return -sm

      def objective_penalty(params, t_init=0.3, penalty=100.0):
          """
          Penalty-based NLP: maximize t subject to all distances >= t and <= 1.
          We minimize -t + penalty * sum(max(0, t - d_i)^2) + penalty * sum(max(0, d_i - 1)^2)
          """
          pts = params[:2*n].reshape((n, 2))
          t = params[2*n]
          
          dists = pdist(pts, 'euclidean')
          
          # Penalty for distances below t
          below_t = np.maximum(0, t - dists)
          # Penalty for distances above 1
          above_1 = np.maximum(0, dists - 1.0)
          
          penalty_term = penalty * (np.sum(below_t**2) + np.sum(above_1**2))
          
          return -t + penalty_term

      def objective_exact(params):
          """Exact non-smooth objective: minimize -min(dist)/max(dist)"""
          pts = params.reshape((n, 2))
          dists = pdist(pts, 'euclidean')
          d_min = np.min(dists)
          d_max = np.max(dists)
          if d_max < 1e-10: return 1.0
          return -(d_min / d_max)

      best_points = None
      best_score = -1.0

      def update_best(pts):
          nonlocal best_points, best_score
          if pts is None or not np.all(np.isfinite(pts)): return
          if pts.shape != (n, 2): return
          s = get_score(pts)
          if s > best_score:
              best_score = s
              best_points = pts.copy()

      # Generate diverse initializations
      def generate_initializations():
          inits = []
          
          # 1. 4x4 grid
          x = np.arange(4)
          y = np.arange(4)
          xx, yy = np.meshgrid(x, y)
          inits.append(np.column_stack((xx.ravel(), yy.ravel())))
          
          # 2. Hexagonal packing
          hex_pts = []
          for i in range(4):
              for j in range(4):
                  x = i + 0.5 * (j % 2)
                  y = j * np.sqrt(3)/2
                  hex_pts.append([x, y])
          inits.append(np.array(hex_pts))
          
          # 3. Circle
          angles = np.linspace(0, 2*np.pi, n, endpoint=False)
          inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
          
          # 4. Spiral
          t = np.linspace(0, 4*np.pi, n)
          r = np.linspace(0.2, 1.0, n)
          inits.append(np.column_stack([r * np.cos(t), r * np.sin(t)]))
          
          # 5. Two concentric circles
          angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
          angles2 = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 8, endpoint=False)
          c1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
          c2 = np.column_stack([1.0 * np.cos(angles2), 1.0 * np.sin(angles2)])
          inits.append(np.vstack([c1, c2]))
          
          # 6. Random (with fixed seed for reproducibility)
          rng = np.random.RandomState(42)
          inits.append(rng.uniform(-1, 1, size=(n, 2)))
          
          # 7. Perturbed grid
          grid = inits[0].copy()
          grid += rng.uniform(-0.1, 0.1, size=(n, 2))
          inits.append(grid)
          
          # 8. Fibonacci sphere projection to 2D
          phi = (1 + np.sqrt(5)) / 2
          golden_angle = 2 * np.pi / (phi ** 2)
          fib_pts = []
          for i in range(n):
              angle = i * golden_angle
              r = np.sqrt(i / n)
              fib_pts.append([r * np.cos(angle), r * np.sin(angle)])
          inits.append(np.array(fib_pts))
          
          return inits

      initializations = generate_initializations()

      # Phase 1: Global search with Differential Evolution on exact objective
      if time.time() < deadline - 180:
          # Fix first two points to break symmetry and scale invariance
          p0 = [0.0, 0.0]
          p1 = [1.0, 0.0]
          
          def objective_de(params):
              pts = np.array([p0, p1] + [list(params[i:i+2]) for i in range(0, 28, 2)])
              return objective_exact(pts.flatten())
          
          bounds = [(-2, 2)] * 28
          try:
              res = differential_evolution(
                  objective_de, 
                  bounds=bounds, 
                  maxiter=200, 
                  popsize=15, 
                  tol=1e-6, 
                  seed=42,
                  workers=1,
                  polish=True,
                  init='sobol'
              )
              if hasattr(res, 'x') and res.x is not None:
                  pts = np.array([p0, p1] + [list(res.x[i:i+2]) for i in range(0, 28, 2)])
                  update_best(pts)
          except Exception as e:
              print("DE failed:", e)

      # Phase 2: Local refinement from multiple initializations
      for init_idx, init_pts in enumerate(initializations):
          if time.time() > deadline - 60:
              break
          
          # 2a: Refine with scaled soft-min objective (L-BFGS-B)
          if time.time() < deadline - 30:
              try:
                  res = minimize(
                      lambda p: objective_scaled(p, beta=100.0), 
                      init_pts.flatten(), 
                      method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                  )
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass
          
          # 2b: Refine with exact objective (Powell)
          if time.time() < deadline - 20:
              try:
                  res = minimize(
                      objective_exact, 
                      init_pts.flatten(), 
                      method='Powell', 
                      options={'maxiter': 1000, 'ftol': 1e-14, 'xtol': 1e-14}
                  )
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass
          
          # 2c: Penalty-based NLP with auxiliary t
          if time.time() < deadline - 15:
              try:
                  # Initialize t to a reasonable value
                  dists_init = pdist(init_pts, 'euclidean')
                  t_init = np.min(dists_init) * 0.8
                  
                  # Parameters: 2n for points + 1 for t
                  init_params = np.concatenate([init_pts.flatten(), [t_init]])
                  
                  res = minimize(
                      objective_penalty, 
                      init_params, 
                      method='Nelder-Mead', 
                      options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                  )
                  if np.all(np.isfinite(res.x)):
                      pts_from_nlp = res.x[:2*n].reshape((n, 2))
                      update_best(pts_from_nlp)
              except Exception:
                  pass

      # Phase 3: High-precision polish on best found
      if best_points is not None and time.time() < deadline - 10:
          # 3a: Nelder-Mead with very tight tolerance
          try:
              res = minimize(
                  objective_exact, 
                  best_points.flatten(), 
                  method='Nelder-Mead', 
                  options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
              )
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass
          
          # 3b: Powell with tight tolerance
          if time.time() < deadline - 5:
              try:
                  res = minimize(
                      objective_exact, 
                      best_points.flatten(), 
                      method='Powell', 
                      options={'maxiter': 5000, 'ftol': 1e-16, 'xtol': 1e-16}
                  )
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass
          
          # 3c: Final L-BFGS-B on scaled soft-min
          if time.time() < deadline - 2:
              try:
                  res = minimize(
                      lambda p: objective_scaled(p, beta=200.0), 
                      best_points.flatten(), 
                      method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
                  )
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass

      if best_points is None:
          # Absolute fallback
          x = np.arange(4)
          y = np.arange(4)
          xx, yy = np.meshgrid(x, y)
          return np.column_stack((xx.ravel(), yy.ravel()))
          
      return best_points
