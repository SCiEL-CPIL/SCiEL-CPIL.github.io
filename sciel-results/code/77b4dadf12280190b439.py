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

      def smooth_objective(params, beta=50.0):
          """
          Smooth approximation of -min(dist)/max(dist) using log-sum-exp.
          Minimizing this maximizes the ratio.
          """
          pts = params.reshape((n, 2))
          dists = pdist(pts, 'euclidean')
          
          if np.max(dists) < 1e-10:
              return 1.0
          
          # Soft max: log(sum(exp(d))) ≈ max(d)
          # Soft min: -log(sum(exp(-d))) ≈ min(d)
          
          # For numerical stability
          d_max = np.max(dists)
          d_min_val = np.min(dists)
          
          # Smooth max
          exp_d = np.exp(dists - d_max)
          soft_max = d_max + np.log(np.sum(exp_d))
          
          # Smooth min
          exp_neg_d = np.exp(-(dists - d_min_val))
          soft_min = d_min_val - np.log(np.sum(exp_neg_d))
          
          if soft_max < 1e-10:
              return 1.0
          
          return -(soft_min / soft_max)

      def objective_fixed(params):
          # params: 28 values for points 2..15
          p0 = [0.0, 0.0]
          p1 = [1.0, 0.0]
          pts = np.array([p0, p1] + [list(params[i:i+2]) for i in range(0, 28, 2)])
          dists = pdist(pts, 'euclidean')
          d_min = np.min(dists)
          d_max = np.max(dists)
          if d_max < 1e-10: return 1.0 # Maximize ratio -> Minimize -ratio
          return -(d_min / d_max)

      best_points = None
      best_score = -1.0

      def update_best(pts):
          nonlocal best_points, best_score
          if pts is None or not np.all(np.isfinite(pts)): return
          s = get_score(pts)
          if s > best_score:
              best_score = s
              best_points = pts.copy()

      # 1. Global Search with Fixed Scale
      if time.time() < deadline - 60:
          bounds = [(-10, 10)] * 28
          try:
              res = differential_evolution(
                  objective_fixed, 
                  bounds=bounds, 
                  maxiter=200, 
                  popsize=15, 
                  tol=1e-6, 
                  seed=42,
                  workers=1
              )
              if res.success or hasattr(res, 'x'):
                  p0 = [0.0, 0.0]
                  p1 = [1.0, 0.0]
                  pts = np.array([p0, p1] + [list(res.x[i:i+2]) for i in range(0, 28, 2)])
                  update_best(pts)
          except Exception as e:
              print("DE failed:", e)

      # 2. Local Refinement with Smooth Objective
      if best_points is not None and time.time() < deadline - 10:
          def obj_smooth(params):
              return smooth_objective(params, beta=50.0)
          
          try:
              res = minimize(obj_smooth, best_points.flatten(), method='BFGS', 
                           options={'maxiter': 1000, 'gtol': 1e-8})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass
          
          # Also try Powell on exact objective
          def obj_full(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          try:
              res = minimize(obj_full, best_points.flatten(), method='Powell', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12})
              if res.success and np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass

      # 3. Hexagonal Cluster Starting Point (known good geometry)
      if time.time() < deadline - 30:
          # Hexagonal cluster of 19 points with 3 removed for better compactness
          hex_pts = np.array([
              [0, 0], [1, 0], [2, 0], [3, 0],
              [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
              [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
              [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598],
              [-0.5, 0.866], [-0.5, 1.732], [4, 0.866]
          ])
          # Take first 16 points
          hex_16 = hex_pts[:16].copy()
          update_best(hex_16)
          
          # Refine hex cluster
          def obj_full2(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          try:
              res = minimize(obj_full2, hex_16.flatten(), method='Powell', 
                           options={'maxiter': 200, 'ftol': 1e-10})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass
          
          # Also try smooth refinement on hex
          try:
              res = minimize(obj_smooth, hex_16.flatten(), method='BFGS', 
                           options={'maxiter': 500, 'gtol': 1e-8})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass

      # 4. Final high-precision polish
      if best_points is not None and time.time() < deadline - 5:
          def obj_highprec(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          try:
              res = minimize(obj_highprec, best_points.flatten(), method='Powell', 
                           options={'maxiter': 1000, 'ftol': 1e-15, 'xtol': 1e-15})
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
